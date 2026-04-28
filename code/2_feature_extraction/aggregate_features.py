# aggregate_features.py
import os
import pandas as pd


def aggregate_features(
    sinkhole_position,
    df_path,
    output_folder_path,
    ssp=None,
):
    """
    汇总所有已提取的特征，保存为一个总表 CSV。

    假设：
    ----
    - 需求1~10 已经依次运行完毕，对同一个 sinkhole_position 不断追加特征列；
    - sinkhole_position 至少包含原始列：
        ['No', 'Disaster', 'Longitude', 'Latitude', 'ADCODE99', 'Province']

    参数
    ----
    sinkhole_position : pandas.DataFrame
        已包含所有特征列的总 DataFrame。
    df_path : str
        主代码中读取的原始 CSV 路径，用于从中提取文件名，拼到输出文件名中。
    output_folder_path : str
        主代码根据 csv_to_subfolder 映射得到的目录：
        D:\\path\\to\\sinkhole-risk-china\\data\\Extracted_HAVE_future\\<csv_to_subfolder[csv_name]>
        本函数会在该目录下保存汇总结果。
    ssp : str, 可选
        当前使用的 SSP 情景（如 'ssp1'/'ssp2'/'ssp3'/'ssp5'），
        若提供，则会拼进输出文件名中以便区分不同情景。

    返回
    ----
    pandas.DataFrame
        汇总后的 DataFrame（与输入 sinkhole_position 相同，只是列顺序重新整理）。
    """

    print("\n[AggregateFeatures] 开始汇总所有特征到单一文件 ...")

    # 1. 检查原始必要列是否存在
    # base_cols = ["No", "Disaster", "Longitude", "Latitude", "ADCODE99", "Province"]
    base_cols = ["No", "Longitude", "Latitude"]
    missing = [c for c in base_cols if c not in sinkhole_position.columns]
    if missing:
        raise ValueError(
            f"[AggregateFeatures] sinkhole_position 缺少必要原始列: {missing}"
        )

    # 2. 整理列顺序：原始 6 列在前，其余特征列在后
    all_cols = list(sinkhole_position.columns)
    other_cols = [c for c in all_cols if c not in base_cols]
    ordered_cols = base_cols + other_cols

    sinkhole_ordered = sinkhole_position[ordered_cols].copy()

    # 3. 构造输出文件名
    csv_name = os.path.basename(df_path)              # 如 Positive_Negative_balanced_25366.csv
    csv_stem, _ = os.path.splitext(csv_name)          # -> Positive_Negative_balanced_25366

    # 文件名格式：
    #   AllFeatures_<原始CSV名>.csv
    #   若提供 ssp，则为 AllFeatures_<原始CSV名>_<ssp>.csv
    if ssp is not None and str(ssp).strip() != "":
        ssp_str = str(ssp).replace(" ", "")
        out_filename = f"AllFeatures_{csv_stem}_{ssp_str}.csv"
    else:
        out_filename = f"AllFeatures_{csv_stem}.csv"

    os.makedirs(output_folder_path, exist_ok=True)
    output_path = os.path.join(output_folder_path, out_filename)

    # 4. 保存 CSV
    sinkhole_ordered.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 5. 简单日志
    print("\n[AggregateFeatures] 汇总完成！文件已保存至:")
    print("  ", output_path)
    print(f"[AggregateFeatures] 总样本数: {len(sinkhole_ordered)}")
    print(f"[AggregateFeatures] 总特征数(含原始列): {len(sinkhole_ordered.columns)}")
    print("\n[AggregateFeatures] 列名预览:")
    print(sinkhole_ordered.columns.tolist())
    print("\n[AggregateFeatures] 数据预览:")
    print(sinkhole_ordered.head())

    return sinkhole_ordered
