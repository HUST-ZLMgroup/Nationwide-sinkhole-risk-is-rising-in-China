# distance_to_fault.py
import os
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


def distance_to_fault(sinkhole_position, database_folder_path, output_folder_path):
    """
    计算每个塌陷点到最近断层的距离（米），并将结果保存为 CSV 文件。

    参数
    ----
    sinkhole_position : pandas.DataFrame
        主代码中已经提取好的 DataFrame，至少需要包含列：
        'No', 'Longitude', 'Latitude'
    database_folder_path : str
        主代码中定义的数据库根目录，例如:
        Z:\\jing\\Large_scale\\future_dataset
    output_folder_path : str
        主代码根据 df_path 映射出的输出文件夹路径。

    返回
    ----
    pandas.DataFrame
        在原 DataFrame 基础上新增 'Distance_to_Fault_m' 列后的对象。
    """

    print("\n[Distance_to_fault] 开始计算塌陷点到断层的最近距离...")

    # ---------------- 1. 检查输入列 ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Distance_to_fault] 输入的 sinkhole_position 缺少必要列: '{col}'"
            )

    # ---------------- 2. 设置断层数据路径并读取 ----------------
    fault_path = os.path.join(
        database_folder_path,
        "Geological_factors",
        "Fault",
        "outputs",
        "faults.shp",
    )
    if not os.path.exists(fault_path):
        raise FileNotFoundError(
            f"[Distance_to_fault] 未找到断层矢量文件: {fault_path}"
        )

    print(f"[Distance_to_fault] 读取断层数据: {fault_path}")
    fault_gdf = gpd.read_file(fault_path)

    # ---------------- 3. 创建塌陷点 GeoDataFrame ----------------
    sinkhole_df = sinkhole_position.copy()
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(sinkhole_df["Longitude"], sinkhole_df["Latitude"])
    ]
    sinkhole_gdf = gpd.GeoDataFrame(sinkhole_df, geometry=geometry, crs="EPSG:4326")

    # ---------------- 4. 处理断层数据的坐标系 ----------------
    if fault_gdf.crs is None:
        print(
            "[Distance_to_fault] 警告：断层数据没有定义坐标系，手动设置为 EPSG:4326 (WGS84)"
        )
        fault_gdf.crs = "EPSG:4326"

    # 若断层与点的 CRS 不一致，先统一到 WGS84
    if fault_gdf.crs != sinkhole_gdf.crs:
        print(
            f"[Distance_to_fault] 将断层数据从 {fault_gdf.crs} 重投影到 EPSG:4326 (WGS84)..."
        )
        fault_gdf = fault_gdf.to_crs(sinkhole_gdf.crs)

    # ---------------- 5. 投影到等距投影坐标系（米） ----------------
    # 使用适合中国区域的投影：EPSG:4479 - China Albers Equal Area Conic
    projected_crs = "EPSG:4479"
    print(f"[Distance_to_fault] 投影断层与塌陷点数据到 {projected_crs} ...")

    fault_gdf_projected = fault_gdf.to_crs(projected_crs)
    sinkhole_gdf_projected = sinkhole_gdf.to_crs(projected_crs)

    # ---------------- 6. 计算点到最近断层距离（米） ----------------
    print("[Distance_to_fault] 开始计算到最近断层的距离（米）...")

    # 将所有断层几何合并，减少距离计算量
    fault_union = fault_gdf_projected.geometry.unary_union

    distances_meters = []
    for geom in tqdm(
        sinkhole_gdf_projected.geometry,
        total=len(sinkhole_gdf_projected),
        desc="计算到最近断层的距离",
    ):
        try:
            dist_val = geom.distance(fault_union)
            distances_meters.append(float(dist_val))
        except Exception as e:
            # 出错时记录 NaN
            distances_meters.append(np.nan)

    if len(distances_meters) != len(sinkhole_position):
        raise RuntimeError(
            "[Distance_to_fault] 计算得到的距离数量与输入点数量不一致，"
            f"points={len(sinkhole_position)}, distances={len(distances_meters)}。"
        )

    # ---------------- 7. 将距离写回 DataFrame ----------------
    sinkhole_position["Distance_to_Fault_m"] = distances_meters

    # ---------------- 8. 保存结果 ----------------
    result_cols = ["No", "Longitude", "Latitude", "Distance_to_Fault_m"]
    result_cols = [c for c in result_cols if c in sinkhole_position.columns]
    result_df = sinkhole_position[result_cols].copy()

    output_path = os.path.join(output_folder_path, "Distance_to_Fault.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. 打印统计信息 ----------------
    print("\n[Distance_to_fault] 处理完成！结果已保存至:")
    print(" ", output_path)
    print(f"[Distance_to_fault] 总塌陷点数: {len(result_df)}")

    if result_df["Distance_to_Fault_m"].notna().any():
        print(
            f"[Distance_to_fault] 平均到断层距离: "
            f"{result_df['Distance_to_Fault_m'].mean():.2f} 米"
        )
        print(
            f"[Distance_to_fault] 最小到断层距离: "
            f"{result_df['Distance_to_Fault_m'].min():.2f} 米"
        )
        print(
            f"[Distance_to_fault] 最大到断层距离: "
            f"{result_df['Distance_to_Fault_m'].max():.2f} 米"
        )

    print("\n[Distance_to_fault] 结果预览:")
    print(result_df.head())

    return sinkhole_position
