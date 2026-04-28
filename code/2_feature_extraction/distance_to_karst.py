# distance_to_karst.py
import os
import geopandas as gpd
from shapely.geometry import Point


def distance_to_karst(sinkhole_position, database_folder_path, output_folder_path):
    """
    计算每个塌陷点到最近碳酸盐岩岩溶区的距离，并将结果保存为 CSV。

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
        在原 DataFrame 基础上新增 'Distance_to_karst' 列后的对象。
    """

    print("\n[Distance_to_karst] 开始计算塌陷点到岩溶区的最近距离...")

    # ---------------- 1. 检查输入列 ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Distance_to_karst] 输入的 sinkhole_position 缺少必要列: '{col}'"
            )

    # ---------------- 2. 读取岩溶区矢量数据 ----------------
    karst_path = os.path.join(
        database_folder_path,
        "Geological_factors",
        "Karst_region",
        "中国岩溶分布_ExportFeatures.shp",
    )
    print(f"[Distance_to_karst] 读取岩溶区数据: {karst_path}")
    karst_gdf = gpd.read_file(karst_path)

    if "rock_type" not in karst_gdf.columns:
        raise ValueError(
            "[Distance_to_karst] 岩溶区数据中未找到字段 'rock_type'，无法筛选碳酸盐岩区域。"
        )

    # 3. 过滤出碳酸盐岩区域（rock_type 为 1 或 2）
    carbonate_karst = karst_gdf[karst_gdf["rock_type"].isin([1, 2])].copy()
    if carbonate_karst.empty:
        raise ValueError(
            "[Distance_to_karst] 过滤后碳酸盐岩区域为空，请检查 rock_type 字段或筛选条件。"
        )

    print(
        f"[Distance_to_karst] 碳酸盐岩多边形数量: {len(carbonate_karst)}, "
        f"总要素数量: {len(karst_gdf)}"
    )

    # ---------------- 4. 将塌陷点转换为 GeoDataFrame ----------------
    sinkhole_df = sinkhole_position.copy()
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(sinkhole_df["Longitude"], sinkhole_df["Latitude"])
    ]
    sinkhole_gdf = gpd.GeoDataFrame(sinkhole_df, geometry=geometry, crs="EPSG:4326")

    # ---------------- 5. 确保坐标系一致 & 转投影 ----------------
    # 先把 karst 重投影到点的 CRS（WGS84，经纬度）
    if carbonate_karst.crs is None:
        raise ValueError(
            "[Distance_to_karst] 岩溶区数据缺少 CRS 信息，请在 shp 中定义坐标系。"
        )

    if carbonate_karst.crs != sinkhole_gdf.crs:
        carbonate_karst = carbonate_karst.to_crs(sinkhole_gdf.crs)

    # 再一起转到等距投影坐标系（UTM 或 Web Mercator），以米作为距离单位
    try:
        utm_crs = sinkhole_gdf.estimate_utm_crs()
        print(f"[Distance_to_karst] 使用估算 UTM 投影: {utm_crs}")
        sinkhole_gdf_utm = sinkhole_gdf.to_crs(utm_crs)
        carbonate_karst_utm = carbonate_karst.to_crs(utm_crs)
    except Exception as e:
        print(
            "[Distance_to_karst] 估算 UTM 投影失败，改用 EPSG:3857 (Web Mercator)。"
        )
        print(f"[Distance_to_karst] 具体错误信息: {e}")
        sinkhole_gdf_utm = sinkhole_gdf.to_crs("EPSG:3857")
        carbonate_karst_utm = carbonate_karst.to_crs("EPSG:3857")

    # ---------------- 6. 计算最近距离（单位：米） ----------------
    print("[Distance_to_karst] 计算点到碳酸盐岩区域的最近距离...")

    # 将所有碳酸盐岩区域合并为一个几何体，减少计算量
    karst_union = carbonate_karst_utm.geometry.unary_union

    # 对每个点计算到合并后多边形的距离
    distances = sinkhole_gdf_utm.geometry.apply(lambda geom: geom.distance(karst_union))

    # ---------------- 7. 将距离写回原 DataFrame ----------------
    sinkhole_position["Distance_to_karst"] = distances.values

    # ---------------- 8. 整理并保存结果 ----------------
    # 这里只输出必要列，如果你后续需要别的列，可以自行在主代码中合并
    result_columns = ["No", "Longitude", "Latitude", "Distance_to_karst"]
    result_columns = [c for c in result_columns if c in sinkhole_position.columns]
    final_result = sinkhole_position[result_columns].copy()

    output_path = os.path.join(output_folder_path, "Distance_to_karst.csv")
    final_result.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. 打印统计信息 ----------------
    print("\n[Distance_to_karst] 处理完成！结果已保存至:")
    print(" ", output_path)
    print(f"[Distance_to_karst] 总塌陷点数: {len(final_result)}")

    if len(final_result) > 0:
        print("\n[Distance_to_karst] 距离统计（米）：")
        print(f"  最小距离: {final_result['Distance_to_karst'].min():.2f}")
        print(f"  最大距离: {final_result['Distance_to_karst'].max():.2f}")
        print(f"  平均距离: {final_result['Distance_to_karst'].mean():.2f}")

        print("\n[Distance_to_karst] 结果预览:")
        print(final_result.head())

    return sinkhole_position
