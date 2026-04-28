import os
import numpy as np
import pandas as pd


def _find_long_nan_runs(series, max_allowed=10):
    """
    查找 series 中连续 NaN 段，返回列表：
    [(start_pos, end_pos, length), ...]，仅保留 length > max_allowed 的段。
    这里的 start_pos/end_pos 是基于当前 Series 的位置索引（0, 1, 2, ...）。
    """
    is_na = series.isna().to_numpy()
    n = len(is_na)
    runs = []

    i = 0
    while i < n:
        if is_na[i]:
            start = i
            while i < n and is_na[i]:
                i += 1
            end = i - 1
            length = end - start + 1
            if length > max_allowed:
                runs.append((start, end, length))
        else:
            i += 1

    return runs


def _normalize_integer_column(series, col_name):
    """
    将指定列统一规范为整数编码。

    处理逻辑：
    1. 先转为数值；
    2. 对非整数值进行四舍五入；
    3. 保留缺失值，最终转为 pandas 可空整数类型 Int64。
    """
    s_num = pd.to_numeric(series, errors="coerce")

    mask_valid = s_num.notna()
    mask_non_integer = mask_valid & (~np.isclose(s_num, np.round(s_num), atol=1e-8))
    n_non_integer = int(mask_non_integer.sum())

    if n_non_integer > 0:
        print(
            f"[DataClean] 检测到 {col_name} 中有 {n_non_integer} 个非整数值，已执行四舍五入取整。"
        )

    s_rounded = s_num.round()
    return pd.Series(pd.array(s_rounded, dtype="Int64"), index=series.index)


def data_clean_and_shuffle(
    sinkhole_position,
    df_path,
    output_folder_path,
    ssp=None,
    random_state=42,
):
    """
    需求12：数据清洗 + 打乱

    步骤：
    1. 针对每一个特征列，删除该特征中超出 mean ± 10*std 范围的行；
    2. 针对每一个特征列，对空值所在的行进行插值填充；
       若某列存在连续 NaN 长度 > 10，则打印出具体信息（不强制报错中断）；
    3. 检查每一个特征列是否存在占比 > 10% 的数值，对这些值随机加减极小数；
    4. 数据整体打乱（按行），重置 index；
    5. 保存到 output_folder_path，文件名 = 需求11 的文件名 + "_cleaned".

    参数
    ----
    sinkhole_position : pandas.DataFrame
        需求11 汇总完的总 DataFrame（包含原始 6 列 + 所有特征列）。
    df_path : str
        原始样本 CSV 路径（主代码中读入的 df_path），用于构造输出文件名。
    output_folder_path : str
        路径为主程序根据 csv_to_subfolder 映射得到的目录。
        即：D:\\path\\to\\sinkhole-risk-china\\data\\Extracted_HAVE_future\\<csv_to_subfolder[csv_name]>
    ssp : str or None
        当前 SSP 情景（如 'ssp1', 'ssp2', 'ssp3', 'ssp5'）。需要和需求11保持一致。
    random_state : int or None
        用于控制随机打乱和扰动的随机种子，便于复现。

    返回
    ----
    pandas.DataFrame
        清洗 + 打乱后的 DataFrame。
    """

    print("\n[DataClean] 开始进行数据清洗与打乱 ...")

    # 为了可重复，设置随机种子
    if random_state is not None:
        np.random.seed(random_state)

    # 0. 复制一份，避免修改原来的 sinkhole_position 引用
    df_clean = sinkhole_position.copy()

    # 0.1 规范需要保持整数的列，避免因浮点精度残留变成非整数
    if "ADCODE99" in df_clean.columns:
        df_clean["ADCODE99"] = _normalize_integer_column(df_clean["ADCODE99"], "ADCODE99")
    if "Disaster" in df_clean.columns:
        df_clean["Disaster"] = _normalize_integer_column(df_clean["Disaster"], "Disaster")

    # 1. 定义“原始列”和“特征列”
    # base_cols = ["No", "Disaster", "Longitude", "Latitude", "ADCODE99", "Province"]
    base_cols = ["No", "Longitude", "Latitude"]
    if "ADCODE99" in df_clean.columns:
        base_cols.append("ADCODE99")
    if "Disaster" in df_clean.columns:
        base_cols.append("Disaster")

    for c in base_cols:
        if c not in df_clean.columns:
            raise ValueError(f"[DataClean] DataFrame 缺少基础列: {c}")

    # 只在数值型列里找特征列
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in base_cols]

    print(f"[DataClean] 检测到数值型特征列数量: {len(feature_cols)}")
    print("[DataClean] 特征列列表:")
    print("  ", feature_cols)

    # 1) 针对每一个特征列，删除超出 mean ± 10*std 的行
    print("\n[DataClean] Step 1: 删除极端离群值 (mean ± 10*std)...")
    for col in feature_cols:
        series = df_clean[col].astype("float64")

        mu = series.mean(skipna=True)
        sigma = series.std(skipna=True)

        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0:
            # 该列基本常数或全 NaN，跳过离群值删除
            print(f"[DataClean] 列 {col}: std 无效或为 0，跳过离群值检测。")
            continue

        lower = mu - 10.0 * sigma
        upper = mu + 10.0 * sigma

        mask_outlier = (series < lower) | (series > upper)
        n_outliers = int(mask_outlier.sum())

        if n_outliers > 0:
            print(
                f"[DataClean] 列 {col}: 删除 {n_outliers} 条超出 [{lower:.4g}, {upper:.4g}] 的离群样本。"
            )
            df_clean = df_clean.loc[~mask_outlier].copy()
        else:
            print(f"[DataClean] 列 {col}: 无离群值。")

    # 删除完行之后，重置 index
    df_clean.reset_index(drop=True, inplace=True)
    print(f"[DataClean] 离群值删除后，样本数: {len(df_clean)}")

    # 2) 对空值进行插值，并检查连续 NaN > 10 的情况
    print("\n[DataClean] Step 2: 插值填补缺失值 + 检查长 NaN 段 ...")
    for col in feature_cols:
        s = df_clean[col].astype("float64")
        if not s.isna().any():
            # 没有缺失，跳过
            continue

        # 2.1 检查是否有连续 NaN 超过 10 个
        long_runs = _find_long_nan_runs(s, max_allowed=10)
        if long_runs:
            print(f"[DataClean][警告] 列 {col} 存在连续超过 10 个的 NaN 段：")
            for start, end, length in long_runs:
                print(
                    f"    - 位置区间: [{start}, {end}]，长度: {length} （基于当前 DataFrame 行序）"
                )

        # 2.2 做线性插值，双向填充（两端也填）
        s_interp = s.interpolate(method="linear", limit_direction="both")
        df_clean[col] = s_interp

        # 再检查是否还有 NaN（理论上连续长段两端都为 NaN 时仍会有残留）
        remaining_nan = df_clean[col].isna().sum()
        if remaining_nan > 0:
            print(
                f"[DataClean][提示] 列 {col} 插值后仍有 {remaining_nan} 个 NaN，"
                f"建议后续手动检查这些位置。"
            )

    # 3) 检查每个特征列是否存在占比 > 10% 的数值，若有则添加微小随机扰动
    print("\n[DataClean] Step 3: 检查并扰动占比 > 10% 的重复值 ...")
    for col in feature_cols:
        s = df_clean[col].astype("float64")
        non_na = s.dropna()

        if len(non_na) == 0:
            print(f"[DataClean] 列 {col}: 全为 NaN，跳过。")
            continue

        value_ratio = non_na.value_counts(normalize=True)
        # 选出占比 > 10% 的所有数值
        heavy_vals = value_ratio[value_ratio > 0.10]

        if heavy_vals.empty:
            print(f"[DataClean] 列 {col}: 不存在占比 >10% 的单一数值。")
            continue

        # 计算一个极小的 eps，相对于该列的尺度
        std_col = non_na.std(skipna=True)
        if not np.isfinite(std_col) or std_col == 0:
            base_scale = 1.0
        else:
            base_scale = std_col

        eps = base_scale * 1e-6
        if eps == 0:
            eps = 1e-6

        print(
            f"[DataClean] 列 {col}: 存在 {len(heavy_vals)} 个值占比 >10%，将为这些值添加随机扰动 (±{eps:.3e})。"
        )

        for val, ratio in heavy_vals.items():
            mask_val = s == val
            n_val = int(mask_val.sum())
            if n_val == 0:
                continue

            noise = (np.random.rand(n_val) - 0.5) * 2.0 * eps  # in [-eps, eps]
            s.loc[mask_val] = s.loc[mask_val].to_numpy() + noise

            print(
                f"    - 值 {val} 占比 {ratio*100:.2f}% ，扰动样本数: {n_val}"
            )

        df_clean[col] = s

    # 4) 数据打乱（行顺序随机）
    print("\n[DataClean] Step 4: 打乱样本顺序 ...")
    df_clean = df_clean.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )
    print(f"[DataClean] 打乱后样本数: {len(df_clean)}")

    # 4.1 保存前再次兜底规范整数列，确保导出结果仍为整数编码
    if "ADCODE99" in df_clean.columns:
        df_clean["ADCODE99"] = _normalize_integer_column(df_clean["ADCODE99"], "ADCODE99")
    if "Disaster" in df_clean.columns:
        df_clean["Disaster"] = _normalize_integer_column(df_clean["Disaster"], "Disaster")

    # 5) 生成输出文件名（需求11文件名 + _cleaned）
    csv_name = os.path.basename(df_path)         # 如 Positive_Negative_balanced_25366.csv
    csv_stem, _ = os.path.splitext(csv_name)     # Positive_Negative_balanced_25366

    # 需求11 的文件名逻辑（与 aggregate_features 保持一致）：
    #   AllFeatures_<csv_stem>.csv 或 AllFeatures_<csv_stem>_<ssp>.csv
    if ssp is not None and str(ssp).strip() != "":
        base_filename = f"AllFeatures_{csv_stem}_{ssp}.csv"
    else:
        base_filename = f"AllFeatures_{csv_stem}.csv"

    # 需求12 输出：在此基础上加上 _cleaned
    clean_filename = base_filename.replace(".csv", "_cleaned.csv")

    os.makedirs(output_folder_path, exist_ok=True)
    output_path = os.path.join(output_folder_path, clean_filename)

    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n[DataClean] 清洗 + 打乱 完成！")
    print("[DataClean] 输出文件路径:")
    print("  ", output_path)
    print(f"[DataClean] 最终样本数: {len(df_clean)}")
    print(f"[DataClean] 最终特征数量(含原始列): {len(df_clean.columns)}")

    return df_clean
