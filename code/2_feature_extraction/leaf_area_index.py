# leaf_area_index.py
import os
import numpy as np
from rasterio_compat import rasterio
from tqdm import tqdm


def _extract_lai_values_for_month(tif_path, sinkhole_position, label=None):
    """
    从给定的 GeoTIFF 中按点提取叶面积指数像元值，返回一个长度与 sinkhole_position 相同的 numpy 数组。
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"[LeafAreaIndex] 未找到栅格文件: {tif_path}")

    desc = f"提取 {label}" if label is not None else f"提取 {os.path.basename(tif_path)}"
    values = []

    with rasterio.open(tif_path) as src:
        nodata = src.nodata

        for _, row in tqdm(
            sinkhole_position.iterrows(),
            total=len(sinkhole_position),
            desc=desc,
        ):
            lon = row["Longitude"]
            lat = row["Latitude"]

            try:
                # 对于 EPSG:4326，index(lon, lat)
                row_idx, col_idx = src.index(lon, lat)
            except Exception:
                values.append(np.nan)
                continue

            # 判断是否在栅格范围内
            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                window = rasterio.windows.Window(col_idx, row_idx, 1, 1)
                data = src.read(1, window=window)
                v = float(data[0][0])

                # 处理 NoData
                if nodata is not None and (v == nodata or np.isclose(v, nodata)):
                    v = np.nan

                values.append(v)
            else:
                values.append(np.nan)

    return np.array(values, dtype="float64")


def leaf_area_index(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """
    提取叶面积指数 (Leaf Area Index, LAI)，按年和时间段计算平均值。

    数据分辨率：0.05° (~2.5 arc-min)，月度数据（每年 12 期）。

    数据路径
    --------
    - 2015 年以前（不含 2015）历史数据：
      Z:\\jing\\Large_scale\\future_dataset\\10_leaf_area_1983_2100_0.05d\\historical\\lai_historical_2xxx_y.tif
      其中 2xxx 为年份，y 为月份 (1-12)

    - 2015 年及以后（含 2015）未来情景数据：
      Z:\\jing\\Large_scale\\future_dataset\\10_leaf_area_1983_2100_0.05d\\sspxxx\\lai_sspxxx_2xxx_y.tif

      其中 sspxxx 与 ssp 变量映射关系（忽略大小写）为：
        ssp=ssp1 -> ssp126
        ssp=ssp2 -> ssp245
        ssp=ssp3 -> ssp370
        ssp=ssp5 -> ssp585

    计算规则
    --------
    1) 每一年先用 3 个代表月份的平均：4 月、8 月、12 月
       年平均 LAI = mean(4 月, 8 月, 12 月)

    2) 历史数据：
       使用 2000, 2010, 2020 三年的年平均 LAI，再求平均：
       LAI_hist_2000_2010_2020

       其中 2000, 2010 来自 historical 路径；
            2020 来自对应 SSP 路径（2015 年及以后走 SSP 数据）

    3) 未来数据：
       以 20 年为时间段、10 年间隔年份做平均：
       - 2020-2040：使用 2020, 2030, 2040 的年平均
       - 2040-2060：使用 2040, 2050, 2060 的年平均
       - 2060-2080：使用 2060, 2070, 2080 的年平均
       - 2080-2100：使用 2080, 2090, 2100 的年平均

       对应输出列：
       LAI_2020_2040, LAI_2040_2060, LAI_2060_2080, LAI_2080_2100

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
        - LAI_hist_2000_2010_2020
        - LAI_2020_2040
        - LAI_2040_2060
        - LAI_2060_2080
        - LAI_2080_2100
    """

    print("\n[LeafAreaIndex] 开始计算叶面积指数 (LAI) ...")

    # ---------------- 1. 检查输入列 ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[LeafAreaIndex] 输入的 sinkhole_position 缺少必要列: '{col}'"
            )

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
            f"[LeafAreaIndex] 不支持的 SSP 情景: {ssp}，当前只支持 {list(ssp_to_sspxxx.keys())}"
        )

    sspxxx = ssp_to_sspxxx[ssp_key]  # 如 'ssp3' -> 'ssp370'

    # ---------------- 3. 构造数据路径根目录 ----------------
    lai_root = os.path.join(
        database_folder_path,
        "10_leaf_area_1983_2100_0.05d",
    )

    # 历史：...\\historical\\lai_historical_2xxx_y.tif
    hist_dir = os.path.join(lai_root, "historical")

    # 未来：...\\sspxxx\\lai_sspxxx_2xxx_y.tif
    # 注意：文件和文件夹均忽略大小写，这里统一使用小写
    ssp_dir = os.path.join(lai_root, sspxxx)

    print(f"[LeafAreaIndex] 历史数据目录: {hist_dir}")
    print(f"[LeafAreaIndex] SSP 投影数据目录: {ssp_dir}")
    print(f"[LeafAreaIndex] 当前 SSP 情景: {ssp} -> {sspxxx}")

    # ---------------- 4. 定义需要的年份与时间窗 ----------------
    # 历史：2000, 2010, 2020
    hist_years = [2000, 2010, 2020]

    # 未来时间窗：2020-2040；2040-2060；2060-2080；2080-2100
    future_windows = {
        "LAI_2020_2040": [2020, 2030, 2040],
        "LAI_2040_2060": [2040, 2050, 2060],
        "LAI_2060_2080": [2060, 2070, 2080],
        "LAI_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for years in future_windows.values() for y in years})
    years_needed = sorted(set(hist_years + all_future_years))

    # 每年的代表月份（4, 8, 12）
    rep_months = [4, 8, 12]

    # ---------------- 5. 按“年”计算：每年 = 三个代表月份的平均 ----------------
    year_values = {}

    for year in years_needed:
        month_arrays = []
        for month in rep_months:
            # 按年份决定使用历史还是 SSP 路径
            if year < 2015:
                tif_name = f"lai_historical_{year}_{month}.tif"
                tif_path = os.path.join(hist_dir, tif_name)
            else:
                # 2015 及以后走 SSP 路径：lai_sspxxx_2xxx_y.tif
                # 如：lai_ssp370_2020_4.tif
                tif_name = f"lai_{sspxxx}_{year}_{month}.tif"
                tif_path = os.path.join(ssp_dir, tif_name)

            label = f"{year}-M{month}"
            print(f"[LeafAreaIndex] 年 {year} 月 {month} 使用的栅格: {tif_path}")
            arr = _extract_lai_values_for_month(tif_path, sinkhole_position, label=label)
            month_arrays.append(arr)

        # 当前年的 LAI = 4/8/12 三个月的平均
        stack = np.vstack(month_arrays)
        year_mean = np.nanmean(stack, axis=0)
        year_values[year] = year_mean

    # ---------------- 6. 历史：2000, 2010, 2020 年年均的平均 ----------------
    print("[LeafAreaIndex] 计算历史平均（2000, 2010, 2020 的年均 LAI）...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "LAI_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------------- 7. 未来各时间段内的平均 ----------------
    for col_name, years in future_windows.items():
        print(f"[LeafAreaIndex] 计算未来时间段 {col_name} 对应年份 {years} 的年均 LAI 平均...")
        stack = np.vstack([year_values[y] for y in years])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------------- 8. 保存历史与未来结果 ----------------
    # 历史结果：ID + 坐标 + 历史平均
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()

    os.makedirs(historical_folder_path, exist_ok=True)
    hist_output_path = os.path.join(
        historical_folder_path, "LeafAreaIndex_historical_2000_2010_2020.csv"
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    # 未来结果：ID + 坐标 + 四个时间段平均
    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()

    os.makedirs(future_ssp_folder_path, exist_ok=True)
    future_output_path = os.path.join(
        future_ssp_folder_path, f"LeafAreaIndex_future_{ssp}.csv"
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. 打印统计信息 ----------------
    print("\n[LeafAreaIndex] 历史数据统计（2000, 2010, 2020 年均 LAI 的平均）:")
    if not hist_df[hist_col].isna().all():
        print(f"  点数: {len(hist_df)}")
        print(f"  最小值: {hist_df[hist_col].min():.4f}")
        print(f"  最大值: {hist_df[hist_col].max():.4f}")
        print(f"  平均值: {hist_df[hist_col].mean():.4f}")

    print("\n[LeafAreaIndex] 未来各时间段统计（按 SSP 情景）:")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f"    点数: {len(col_series)}")
        print(f"    最小值: {col_series.min():.4f}")
        print(f"    最大值: {col_series.max():.4f}")
        print(f"    平均值: {col_series.mean():.4f}")

    print("\n[LeafAreaIndex] 历史结果已保存至:")
    print("  ", hist_output_path)
    print("[LeafAreaIndex] 未来结果已保存至:")
    print("  ", future_output_path)
    print("\n[LeafAreaIndex] 结果预览（历史部分）:")
    print(hist_df.head())
    print("\n[LeafAreaIndex] 结果预览（未来部分）:")
    print(future_df.head())

    return sinkhole_position
