# tasmax.py
import os
import numpy as np
import xarray as xr


def _extract_tasmax_for_year(nc_path, sinkhole_position, label=None):
    """
    从给定的 NetCDF 文件中提取每个点该年的平均最高气温（对 time 维度取平均）。

    返回值为长度与 sinkhole_position 相同的一维 numpy 数组。
    若变量单位为 K / Kelvin，则转换为 ℃。
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"[Tasmax] 未找到 NetCDF 文件: {nc_path}")

    print(f"[Tasmax] 打开 {label or os.path.basename(nc_path)}: {nc_path}")

    ds = xr.open_dataset(nc_path)
    try:
        # 1. 选择最高气温变量：优先使用 'tasmax'
        if "tasmax" in ds.data_vars:
            da = ds["tasmax"]
        else:
            # 如果没有 tasmax，就取第一个变量（一般不会发生，只是兜底）
            first_var = list(ds.data_vars)[0]
            da = ds[first_var]
            print(f"[Tasmax] 警告: 未找到变量 'tasmax'，使用 {first_var}")

        # 2. 对 time 维度求平均（年平均）
        time_dim_candidates = [d for d in da.dims if "time" in d.lower()]
        if time_dim_candidates:
            time_dim = time_dim_candidates[0]
            da_mean = da.mean(dim=time_dim)
        else:
            # 若没有 time 维度，则视为已经是年平均
            da_mean = da

        # 3. 找到纬度/经度维度名称，并统一改名为 lat / lon 方便插值
        lat_name = next((d for d in da_mean.dims if "lat" in d.lower()), None)
        lon_name = next((d for d in da_mean.dims if "lon" in d.lower()), None)
        if lat_name is None or lon_name is None:
            raise ValueError(
                "[Tasmax] 无法在 NetCDF 中识别纬度/经度维度，请检查文件结构。"
            )

        rename_dict = {}
        if lat_name != "lat":
            rename_dict[lat_name] = "lat"
        if lon_name != "lon":
            rename_dict[lon_name] = "lon"
        da_mean = da_mean.rename(rename_dict)

        # 4. 使用最近邻插值，将栅格映射到塌陷点经纬度
        lats = sinkhole_position["Latitude"].values
        lons = sinkhole_position["Longitude"].values

        pts_lat = xr.DataArray(lats, dims="points")
        pts_lon = xr.DataArray(lons, dims="points")

        # 最近邻插值
        sample = da_mean.sel(lat=pts_lat, lon=pts_lon, method="nearest")
        vals = sample.values.astype("float64")

        # 5. 单位转换：若为 K / Kelvin，则转换为 ℃
        units_raw = str(da.attrs.get("units", "")).strip().lower()
        units_norm = units_raw.replace(" ", "")
        if (
            units_norm in {"k", "degk", "degreekelvin", "degreeskelvin"}
            or "kelvin" in units_norm
        ):
            vals = vals - 273.15

        # 6. 将非有限值设置为 NaN
        vals = np.where(np.isfinite(vals), vals, np.nan)

        return vals
    finally:
        ds.close()



def tasmax(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """
    计算最高气温（年平均），并在历史/未来时间段取多年月平均。

    数据路径
    --------
    分辨率、时间组织方式与降水数据保持一致，文件按年份划分。

    - 2015 年以前（不含 2015）的历史数据示例：
      Z:\\jing\\Large_scale\\future_dataset\\21_tasmax\\historical\\
      tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_2000_v2.0.nc

      假定其他年份文件名模式：
      tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_{year}_v2.0.nc

    - 2015 年及以后（含 2015）的 SSP 数据示例：
      Z:\\jing\\Large_scale\\future_dataset\\21_tasmax\\sspxxx\\
      tasmax_day_BCC-CSM2-MR_sspxxx_r1i1p1f1_gn_2017_v2.0.nc

      假定其他年份文件名模式：
      tasmax_day_BCC-CSM2-MR_{sspxxx}_r1i1p1f1_gn_{year}_v2.0.nc

      其中 sspxxx 与 ssp 变量映射关系（忽略大小写）为：
        ssp=ssp1 -> ssp126
        ssp=ssp2 -> ssp245
        ssp=ssp3 -> ssp370
        ssp=ssp5 -> ssp585

    计算规则
    --------
    1) 对每个年文件，先对 time 维进行平均，得到该年的年平均最高气温场。

    2) 历史数据：
       使用 2000, 2010, 2020 三个年份的年平均最高气温，再求平均：
       Tasmax_hist_2000_2010_2020

       其中：
         - 2000, 2010 来自 historical 目录
         - 2020 来自对应的 SSP 目录（2015 年及以后走 SSP）

    3) 未来数据：
       以 20 年为时间段，使用 10 年间隔年份的年平均最高气温，三年平均：
       - 2020-2040：使用 2020, 2030, 2040
       - 2040-2060：使用 2040, 2050, 2060
       - 2060-2080：使用 2060, 2070, 2080
       - 2080-2100：使用 2080, 2090, 2100

       对应输出列：
       Tasmax_2020_2040, Tasmax_2040_2060, Tasmax_2060_2080, Tasmax_2080_2100

    参数
    ----
    sinkhole_position : pandas.DataFrame
        至少包含列 'No', 'Longitude', 'Latitude'
    database_folder_path : str
        数据库根目录，例如 Z:\\jing\\Large_scale\\future_dataset
    historical_folder_path : str
        历史数据输出目录（.../historical）
    future_ssp_folder_path : str
        未来数据输出目录（.../future/sspX）
    ssp : str
        SSP 情景字符串，例如 'ssp1', 'ssp2', 'ssp3', 'ssp5'

    返回
    ----
    pandas.DataFrame
        在原 DataFrame 基础上新增以下列：
        - Tasmax_hist_2000_2010_2020
        - Tasmax_2020_2040
        - Tasmax_2040_2060
        - Tasmax_2060_2080
        - Tasmax_2080_2100

    说明
    ----
    若原始 tasmax 单位为 K / Kelvin，则结果自动转换为 ℃；
    若原始单位已是 ℃ 或其他单位，则保持原值。
    """

    print("\n[Tasmax] 开始计算最高气温 ...")

    # ---------------- 1. 检查输入列 ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(f"[Tasmax] 输入的 sinkhole_position 缺少必要列: '{col}'")

    # ---------------- 2. SSP -> sspxxx 映射 ----------------
    ssp_to_sspxxx = {
        "ssp1": "ssp126",
        "ssp2": "ssp245",
        "ssp3": "ssp370",
        "ssp5": "ssp585",
    }

    ssp_key = ssp.lower()
    if ssp_key not in ssp_to_sspxxx:
        raise ValueError(
            f"[Tasmax] 不支持的 SSP 情景: {ssp}，当前只支持 {list(ssp_to_sspxxx.keys())}"
        )

    sspxxx = ssp_to_sspxxx[ssp_key]  # 如 'ssp3' -> 'ssp370'

    # ---------------- 3. 构造数据路径根目录 ----------------
    tasmax_root = os.path.join(
        database_folder_path,
        "21_tasmax",
    )

    # 历史 nc 所在目录
    hist_dir = os.path.join(tasmax_root, "historical")

    # SSP nc 所在目录：.../sspxxx/
    ssp_dir = os.path.join(tasmax_root, sspxxx)

    print(f"[Tasmax] 历史数据目录: {hist_dir}")
    print(f"[Tasmax] SSP 投影数据目录: {ssp_dir}")
    print(f"[Tasmax] 当前 SSP 情景: {ssp} -> {sspxxx}")

    # ---------------- 4. 定义需要的年份与时间窗 ----------------
    hist_years = [2000, 2010, 2020]

    future_windows = {
        "Tasmax_2020_2040": [2020, 2030, 2040],
        "Tasmax_2040_2060": [2040, 2050, 2060],
        "Tasmax_2060_2080": [2060, 2070, 2080],
        "Tasmax_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for years in future_windows.values() for y in years})
    years_needed = sorted(set(hist_years + all_future_years))

    # ---------------- 5. 为每个年份提取年平均最高气温 ----------------
    year_values = {}

    for year in years_needed:
        if year < 2015:
            # tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_{year}_v2.0.nc
            nc_name = f"tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_{year}_v2.0.nc"
            nc_path = os.path.join(hist_dir, nc_name)
        else:
            # tasmax_day_BCC-CSM2-MR_{sspxxx}_r1i1p1f1_gn_{year}_v2.0.nc
            nc_name = f"tasmax_day_BCC-CSM2-MR_{sspxxx}_r1i1p1f1_gn_{year}_v2.0.nc"
            nc_path = os.path.join(ssp_dir, nc_name)

        label = f"{year}"
        year_values[year] = _extract_tasmax_for_year(
            nc_path, sinkhole_position, label=label
        )

    # ---------------- 6. 历史：2000, 2010, 2020 年平均再取平均 ----------------
    print("[Tasmax] 计算历史平均（2000, 2010, 2020）...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "Tasmax_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------------- 7. 未来各时间段内的平均 ----------------
    for col_name, years in future_windows.items():
        print(f"[Tasmax] 计算未来时间段 {col_name} 对应年份 {years} 的年平均最高气温平均...")
        stack = np.vstack([year_values[y] for y in years])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------------- 8. 保存历史与未来结果 ----------------
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()

    os.makedirs(historical_folder_path, exist_ok=True)
    hist_output_path = os.path.join(
        historical_folder_path, "Tasmax_historical_2000_2010_2020.csv"
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()

    os.makedirs(future_ssp_folder_path, exist_ok=True)
    future_output_path = os.path.join(
        future_ssp_folder_path, f"Tasmax_future_{ssp}.csv"
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. 打印统计信息 ----------------
    print("\n[Tasmax] 历史数据统计（2000, 2010, 2020 年均最高气温的平均）:")
    if not hist_df[hist_col].isna().all():
        print(f"  点数: {len(hist_df)}")
        print(f"  最小值: {hist_df[hist_col].min():.4f}")
        print(f"  最大值: {hist_df[hist_col].max():.4f}")
        print(f"  平均值: {hist_df[hist_col].mean():.4f}")

    print("\n[Tasmax] 未来各时间段统计（按 SSP 情景）:")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f"    点数: {len(col_series)}")
        print(f"    最小值: {col_series.min():.4f}")
        print(f"    最大值: {col_series.max():.4f}")
        print(f"    平均值: {col_series.mean():.4f}")

    print("\n[Tasmax] 历史结果已保存至:")
    print("  ", hist_output_path)
    print("[Tasmax] 未来结果已保存至:")
    print("  ", future_output_path)
    print("\n[Tasmax] 结果预览（历史部分）:")
    print(hist_df.head())
    print("\n[Tasmax] 结果预览（未来部分）:")
    print(future_df.head())

    return sinkhole_position
