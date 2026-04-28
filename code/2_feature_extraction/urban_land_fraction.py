# urban_land_fraction.py
import os
import numpy as np
from rasterio_compat import rasterio
from tqdm import tqdm


def _extract_raster_values_for_year(tif_path, sinkhole_position, year_label=None):
    """
    从给定的 GeoTIFF 中按点提取像元值，返回一个长度与 sinkhole_position 相同的 numpy 数组。
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"[Urban_land_fraction] 未找到栅格文件: {tif_path}")

    desc = f"提取 {year_label}" if year_label is not None else f"提取 {os.path.basename(tif_path)}"
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


def urban_land_fraction(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """
    计算城市土地与所有土地面积的比例（urban land fraction）：
    - 历史：2000, 2010, 2020 三个时刻的平均；
    - 未来：2020-2040、2040-2060、2060-2080、2080-2100 四个时间段内（10 年间隔数据）的平均。

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
        SSP 情景字符串，例如 'ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5'

    返回
    ----
    pandas.DataFrame
        在原 DataFrame 基础上新增以下列：
        - UrbanFrac_hist_2000_2010_2020
        - UrbanFrac_2020_2040
        - UrbanFrac_2040_2060
        - UrbanFrac_2060_2080
        - UrbanFrac_2080_2100
    """

    print("\n[Urban_land_fraction] 开始计算城市土地与所有土地面积的比例...")

    # ---------------- 1. 检查输入列 ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Urban_land_fraction] 输入的 sinkhole_position 缺少必要列: '{col}'"
            )

    # ---------------- 2. 构造数据路径 ----------------
    # 根目录：3_fraction_of_urban_land_2000_2100_0.125d/UrbanFraction_1_8_dgr_GEOTIFF_v1
    urban_root = os.path.join(
        database_folder_path,
        "3_fraction_of_urban_land_2000_2100_0.125d",
        "UrbanFraction_1_8_dgr_GEOTIFF_v1",
    )

    # 2000 年基准年数据
    base2000_dir = os.path.join(
        urban_root,
        "UrbanFraction_1_8_dgr_GEOTIFF_BaseYear_2000_v1",
    )
    base2000_path = os.path.join(base2000_dir, "urb_frac_2000.tif")

    # 2010-2100 年 SSP 投影数据
    proj_dir = os.path.join(
        urban_root,
        "UrbanFraction_1_8_dgr_GEOTIFF_Projections_SSPs1-5_2010-2100_v1",
    )

    print(f"[Urban_land_fraction] BaseYear 2000 栅格路径: {base2000_path}")
    print(f"[Urban_land_fraction] SSP 投影数据目录: {proj_dir}")
    print(f"[Urban_land_fraction] 当前 SSP 情景: {ssp}")

    # ---------------- 3. 确定需要的年份 ----------------
    # 历史：2000, 2010, 2020
    hist_years = [2000, 2010, 2020]

    # 未来时间窗：2020-2040；2040-2060；2060-2080；2080-2100
    # 数据按 10 年间隔，窗口内取平均
    future_windows = {
        "UrbanFrac_2020_2040": [2020, 2030, 2040],
        "UrbanFrac_2040_2060": [2040, 2050, 2060],
        "UrbanFrac_2060_2080": [2060, 2070, 2080],
        "UrbanFrac_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for years in future_windows.values() for y in years})
    years_needed = sorted(set(hist_years + all_future_years))

    # ---------------- 4. 为每个年份提取栅格值 ----------------
    year_values = {}

    for year in years_needed:
        if year == 2000:
            tif_path = base2000_path
        else:
            # 2010-2100: 使用 SSP 对应的投影文件，命名格式：sspx_2xxx.tif
            # 例如 ssp3_2010.tif, ssp3_2020.tif ...
            tif_name = f"{ssp}_{year}.tif"
            tif_path = os.path.join(proj_dir, tif_name)

        print(f"[Urban_land_fraction] 年份 {year} 使用的栅格: {tif_path}")
        year_values[year] = _extract_raster_values_for_year(
            tif_path, sinkhole_position, year_label=str(year)
        )

    # ---------------- 5. 历史：2000, 2010, 2020 平均 ----------------
    print("[Urban_land_fraction] 计算历史平均（2000, 2010, 2020）...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "UrbanFrac_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------------- 6. 未来四个时间段内的平均 ----------------
    for col_name, years in future_windows.items():
        print(f"[Urban_land_fraction] 计算未来时间段 {col_name} 对应年份 {years} 的平均...")
        stack = np.vstack([year_values[y] for y in years])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------------- 7. 保存历史与未来结果 ----------------
    # 历史结果：只输出 ID + 坐标 + 历史平均
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()

    os.makedirs(historical_folder_path, exist_ok=True)
    hist_output_path = os.path.join(
        historical_folder_path, "UrbanFraction_historical_2000_2010_2020.csv"
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    # 未来结果：ID + 坐标 + 四个时间段的平均
    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()

    os.makedirs(future_ssp_folder_path, exist_ok=True)
    future_output_path = os.path.join(
        future_ssp_folder_path, f"UrbanFraction_future_{ssp}.csv"
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------------- 8. 打印统计信息 ----------------
    print("\n[Urban_land_fraction] 历史数据统计（2000, 2010, 2020 平均）:")
    if not hist_df[hist_col].isna().all():
        print(f"  点数: {len(hist_df)}")
        print(f"  最小值: {hist_df[hist_col].min():.4f}")
        print(f"  最大值: {hist_df[hist_col].max():.4f}")
        print(f"  平均值: {hist_df[hist_col].mean():.4f}")

    print("\n[Urban_land_fraction] 未来各时间段统计（按 SSP 情景）:")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f"    点数: {len(col_series)}")
        print(f"    最小值: {col_series.min():.4f}")
        print(f"    最大值: {col_series.max():.4f}")
        print(f"    平均值: {col_series.mean():.4f}")

    print("\n[Urban_land_fraction] 历史结果已保存至:")
    print("  ", hist_output_path)
    print("[Urban_land_fraction] 未来结果已保存至:")
    print("  ", future_output_path)
    print("\n[Urban_land_fraction] 结果预览（历史部分）:")
    print(hist_df.head())
    print("\n[Urban_land_fraction] 结果预览（未来部分）:")
    print(future_df.head())

    return sinkhole_position
