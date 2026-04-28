# depth_to_bedrock.py
import os
import numpy as np
from rasterio_compat import rasterio
from tqdm import tqdm


def depth_to_bedrock(sinkhole_position, database_folder_path, output_folder_path):
    """
    提取每个塌陷点的基岩深度（Depth_to_Bedrock），并保存为 CSV 文件。

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
        在原 DataFrame 基础上新增 'Depth_to_Bedrock' 列后的对象。
    """

    print("\n[Depth_to_bedrock] 开始提取基岩深度信息...")

    # ---------------- 1. 检查输入列 ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Depth_to_bedrock] 输入的 sinkhole_position 缺少必要列: '{col}'"
            )

    # ---------------- 2. 设置基岩深度 TIFF 路径 ----------------
    dtb_path = os.path.join(
        database_folder_path,
        "Geological_factors",
        "Depth_to_bedrock",
        "DTB_CHINA_1k.tif",
    )
    if not os.path.exists(dtb_path):
        raise FileNotFoundError(
            f"[Depth_to_bedrock] 未找到基岩深度栅格文件: {dtb_path}"
        )

    print(f"[Depth_to_bedrock] 读取基岩深度栅格: {dtb_path}")

    dtb_values = []

    # ---------------- 3. 使用 rasterio 打开 TIFF 并逐点提取 ----------------
    with rasterio.open(dtb_path) as src:
        print(f"[Depth_to_bedrock] 栅格 CRS: {src.crs}")
        print(
            f"[Depth_to_bedrock] 栅格尺寸: 宽度={src.width}, 高度={src.height}, 分辨率={src.res}"
        )

        # tqdm 进度条
        for idx, row in tqdm(
            sinkhole_position.iterrows(),
            total=len(sinkhole_position),
            desc="提取基岩深度",
        ):
            lon = row["Longitude"]
            lat = row["Latitude"]

            try:
                # 将经纬度转换为栅格行列号
                # 注意：src.index(x, y) 返回 (row, col)
                row_idx, col_idx = src.index(lon, lat)
            except Exception:
                # 若转换失败，记为缺失
                dtb_values.append(np.nan)
                continue

            # 判断是否在栅格范围内
            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                window = rasterio.windows.Window(col_idx, row_idx, 1, 1)
                value = src.read(1, window=window)
                dtb_values.append(float(value[0][0]))
            else:
                dtb_values.append(np.nan)

    # ---------------- 4. 写回 DataFrame ----------------
    if len(dtb_values) != len(sinkhole_position):
        raise RuntimeError(
            "[Depth_to_bedrock] 提取的基岩深度数量与输入点数量不一致，"
            f"points={len(sinkhole_position)}, values={len(dtb_values)}。"
        )

    sinkhole_position["Depth_to_Bedrock"] = dtb_values

    # ---------------- 5. 保存结果 ----------------
    output_path = os.path.join(output_folder_path, "Depth_to_Bedrock.csv")
    sinkhole_position.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------------- 6. 打印统计信息 ----------------
    print("\n[Depth_to_bedrock] 处理完成！结果已保存至:")
    print(" ", output_path)
    print(f"[Depth_to_bedrock] 总塌陷点数: {len(sinkhole_position)}")
    print(
        f"[Depth_to_bedrock] 成功提取深度的点数: "
        f"{sinkhole_position['Depth_to_Bedrock'].notna().sum()}"
    )

    if sinkhole_position["Depth_to_Bedrock"].notna().any():
        mean_val = sinkhole_position["Depth_to_Bedrock"].mean()
        print(f"[Depth_to_bedrock] 平均基岩深度: {mean_val:.2f} 米")

    print("\n[Depth_to_bedrock] 结果预览:")
    print(sinkhole_position.head())

    return sinkhole_position
