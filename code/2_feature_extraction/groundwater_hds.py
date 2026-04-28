# groundwater_hds.py
import os
import numpy as np
import xarray as xr
import pandas as pd

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[GroundwaterHDS] 未安装 scipy，将跳过‘最近样本点补值’，只保留原始插值结果。")


def _fill_nans_by_nearest_sample(values, lons, lats, label_prefix="[GroundwaterHDS]"):
    """
    在“塌陷点样本集合”内部，用最近的有值样本来填补 NaN。

    values : 一维数组，长度 = n_points（某一年的 HDS）
    lons, lats : 对应点的经纬度
    返回：填补后的 values（新数组，不在原地修改）
    """
    values = np.asarray(values, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    lats = np.asarray(lats, dtype="float64")

    if not HAS_SCIPY:
        print(f"{label_prefix} 未安装 scipy，跳过最近样本点补值，保持 NaN。")
        return values

    mask_valid = np.isfinite(values)
    if mask_valid.all():
        # 没有缺失，无需补
        return values
    if not np.any(mask_valid):
        # 全是 NaN，也没法补
        print(f"{label_prefix} 警告：该年份所有点均为 NaN，无法做最近点补值。")
        return values

    coords_valid = np.column_stack([lons[mask_valid], lats[mask_valid]])
    coords_nan = np.column_stack([lons[~mask_valid], lats[~mask_valid]])

    tree = cKDTree(coords_valid)
    dist, idx = tree.query(coords_nan, k=1)

    filled = values.copy()
    filled[~mask_valid] = values[mask_valid][idx]

    return filled


def _extract_hds_from_zarr(store_path, years, sinkhole_position, label_prefix="[GroundwaterHDS]"):
    """
    从给定 Zarr 数据集中，针对指定年份列表，提取每个塌陷点的
    年平均 HDS（Groundwater Head），返回 {year: np.ndarray}。

    步骤：
    1. 对指定年份，从 time 维上取出对应年份并求年平均 -> 2D 场 (lat, lon)
    2. 使用 xarray 的 .sel(lat, lon, method="nearest") 对每个塌陷点插值
    3. 对于插值后仍为 NaN 的点，用“最近有值的样本点”填补（需求 10.1）
    """

    if not years:
        return {}

    if not os.path.exists(store_path):
        raise FileNotFoundError(f"{label_prefix} 未找到 Zarr 数据集: {store_path}")

    print(f"{label_prefix} 打开 Zarr 数据集: {store_path}")

    ds = xr.open_zarr(store_path)
    try:
        # 1. 选 HDS 变量：优先 l1_hds -> hds -> HDS -> l2_hds
        candidate_vars = ["l1_hds", "hds", "HDS", "l2_hds"]
        hds_var = None
        for v in candidate_vars:
            if v in ds.data_vars:
                hds_var = v
                break
        if hds_var is None:
            hds_var = list(ds.data_vars)[0]
            print(f"{label_prefix} 警告: 未找到 l1_hds/l2_hds，使用变量 {hds_var}")

        da = ds[hds_var]

        # 2. 统一经纬度名字：把 latitude/longitude 改成 lat/lon（无论是坐标还是维度）
        rename_dict = {}
        if "latitude" in ds.coords or "latitude" in ds.dims:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds.coords or "longitude" in ds.dims:
            rename_dict["longitude"] = "lon"
        if rename_dict:
            ds = ds.rename(rename_dict)
            da = ds[hds_var]

        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(f"{label_prefix} 数据中缺少 'lat'/'lon' 坐标，请检查 Zarr 结构。")

        # 3. 时间 -> 年份
        if "time" not in ds.coords:
            raise ValueError(f"{label_prefix} 数据集中缺少 'time' 坐标。")

        time_coord = ds["time"]
        tvals = time_coord.values
        if np.issubdtype(tvals.dtype, np.datetime64):
            years_all = pd.to_datetime(tvals).year
        else:
            years_all = np.array(tvals).astype(int)

        # 4. 准备塌陷点坐标
        lats_pts = sinkhole_position["Latitude"].values
        lons_pts = sinkhole_position["Longitude"].values
        pts_lat = xr.DataArray(lats_pts, dims="points")
        pts_lon = xr.DataArray(lons_pts, dims="points")

        result = {}

        for year in years:
            idxs = np.where(years_all == year)[0]
            if idxs.size == 0:
                raise ValueError(
                    f"{label_prefix} 在 {store_path} 中未找到年份 {year} 对应的 time 条目。"
                )

            print(f"{label_prefix} 处理年份 {year} ...")
            da_year = da.isel(time=idxs)
            da_year_mean = da_year.mean(dim="time")

            # 5. 先做标准的最近邻插值
            sample = da_year_mean.sel(lat=pts_lat, lon=pts_lon, method="nearest")
            vals = sample.values.astype("float64")
            vals = np.where(np.isfinite(vals), vals, np.nan)

            # 6. 在“样本集合内部”用最近有值样本补 NaN（需求 10.1）
            vals_filled = _fill_nans_by_nearest_sample(
                vals, lons_pts, lats_pts, label_prefix=f"{label_prefix}-year{year}"
            )

            result[year] = vals_filled

        return result

    finally:
        ds.close()


def groundwater_hds(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """
    地下水埋头 HDS（Groundwater Head）提取与历史/未来时间段平均计算（含 10.1 最近样本点补值）。

    数据结构
    --------
    根目录:
      Z:\\jing\\Large_scale\\future_dataset\\18_groundwater_1960_2100_1km\\Annual

    · 历史 (1960–2014):
      Annual\\historical\\hds_annual_1960_2014_historical_ensemble.zarr

    · 未来 SSP (2015–2100):
      Annual\\sspxxx\\hds_annual_2015_2100_sspxxx_ensemble.zarr

      其中 sspxxx 与 ssp 映射（忽略大小写）为：
        ssp=ssp1 -> ssp126
        ssp=ssp3 -> ssp370
        ssp=ssp5 -> ssp585

    计算规则
    --------
    · 每个年份：使用该年 HDS 年平均值（对 time 维求均值），
      然后对塌陷点插值 + 最近样本点补值。

    · 历史：
        使用 2000, 2010, 2020 三年的 HDS 取平均：
        - 2000, 2010 来自 historical
        - 2020 来自对应 SSP 数据集 (2015+)
        输出列：HDS_hist_2000_2010_2020

    · 未来：
        以 20 年为时间段，以 10 年为间隔取平均：
        - 2020–2040: 2020, 2030, 2040
        - 2040–2060: 2040, 2050, 2060
        - 2060–2080: 2060, 2070, 2080
        - 2080–2100: 2080, 2090, 2100

        输出列：
        HDS_2020_2040, HDS_2040_2060, HDS_2060_2080, HDS_2080_2100
    """

    print("\n[GroundwaterHDS] 开始计算地下水埋头 (HDS) ...")

    # ---------- 1. 检查输入列 ----------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[GroundwaterHDS] 输入的 sinkhole_position 缺少必要列: '{col}'"
            )

    # ---------- 2. SSP -> sspxxx 映射 ----------
    ssp_to_sspxxx = {
        "ssp1": "ssp126",
        "ssp3": "ssp370",
        "ssp5": "ssp585",
    }
    ssp_key = ssp.lower()
    if ssp_key not in ssp_to_sspxxx:
        raise ValueError(
            f"[GroundwaterHDS] 不支持的 SSP 情景: {ssp}，仅支持 {list(ssp_to_sspxxx.keys())}"
        )
    sspxxx = ssp_to_sspxxx[ssp_key]  # 如 'ssp3' -> 'ssp370'

    # ---------- 3. 构造路径 ----------
    gw_root = os.path.join(
        database_folder_path,
        "18_groundwater_1960_2100_1km",
        "Annual",
    )

    # 历史 Zarr
    hist_store = os.path.join(
        gw_root,
        "historical",
        "hds_annual_1960_2014_historical_ensemble.zarr",
    )

    # 未来 SSP Zarr: Annual/sspxxx/hds_annual_2015_2100_sspxxx_ensemble.zarr
    ssp_dir = os.path.join(gw_root, sspxxx)
    ssp_store = os.path.join(
        ssp_dir,
        f"hds_annual_2015_2100_{sspxxx}_ensemble.zarr",
    )

    print(f"[GroundwaterHDS] 历史 Zarr: {hist_store}")
    print(f"[GroundwaterHDS] SSP   Zarr: {ssp_store}")
    print(f"[GroundwaterHDS] 当前 SSP: {ssp} -> {sspxxx}")

    # ---------- 4. 需要的年份 ----------
    hist_years = [2000, 2010, 2020]

    future_windows = {
        "HDS_2020_2040": [2020, 2030, 2040],
        "HDS_2040_2060": [2040, 2050, 2060],
        "HDS_2060_2080": [2060, 2070, 2080],
        "HDS_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for ys in future_windows.values() for y in ys})
    years_needed = sorted(set(hist_years + all_future_years))

    hist_years_store = [y for y in years_needed if y < 2015]   # 用历史 zarr
    ssp_years_store = [y for y in years_needed if y >= 2015]   # 用 SSP zarr

    # ---------- 5. 从 Zarr 提取各年份 HDS ----------
    year_values = {}

    hist_map = _extract_hds_from_zarr(
        hist_store,
        hist_years_store,
        sinkhole_position,
        label_prefix="[GroundwaterHDS-HIST]",
    )
    year_values.update(hist_map)

    ssp_map = _extract_hds_from_zarr(
        ssp_store,
        ssp_years_store,
        sinkhole_position,
        label_prefix="[GroundwaterHDS-SSP]",
    )
    year_values.update(ssp_map)

    missing_years = [y for y in years_needed if y not in year_values]
    if missing_years:
        raise RuntimeError(
            f"[GroundwaterHDS] 以下年份未能从 Zarr 中提取: {missing_years}"
        )

    # ---------- 6. 历史平均 (2000, 2010, 2020) ----------
    print("[GroundwaterHDS] 计算历史平均 (2000, 2010, 2020)...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "HDS_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------- 7. 未来各时间段平均 ----------
    for col_name, ys in future_windows.items():
        print(f"[GroundwaterHDS] 计算未来时间段 {col_name} 对应年份 {ys} 的平均...")
        stack = np.vstack([year_values[y] for y in ys])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------- 8. 保存结果 ----------
    os.makedirs(historical_folder_path, exist_ok=True)
    os.makedirs(future_ssp_folder_path, exist_ok=True)

    # 历史 CSV
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()
    hist_output_path = os.path.join(
        historical_folder_path,
        "GroundwaterHDS_historical_2000_2010_2020.csv",
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    # 未来 CSV
    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()
    future_output_path = os.path.join(
        future_ssp_folder_path,
        f"GroundwaterHDS_future_{ssp}.csv",
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------- 9. 打印简单统计 ----------
    print("\n[GroundwaterHDS] 历史数据统计 (2000, 2010, 2020 的平均):")
    if not hist_df[hist_col].isna().all():
        print(f"  点数: {len(hist_df)}")
        print(f"  最小值: {hist_df[hist_col].min():.4f}")
        print(f"  最大值: {hist_df[hist_col].max():.4f}")
        print(f"  平均值: {hist_df[hist_col].mean():.4f}")

    print("\n[GroundwaterHDS] 未来各时间段统计:")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f"    点数: {len(col_series)}")
        print(f"    最小值: {col_series.min():.4f}")
        print(f"    最大值: {col_series.max():.4f}")
        print(f"    平均值: {col_series.mean():.4f}")

    print("\n[GroundwaterHDS] 历史结果已保存至:")
    print("  ", hist_output_path)
    print("[GroundwaterHDS] 未来结果已保存至:")
    print("  ", future_output_path)

    return sinkhole_position
