# feature_state_io.py
import os
import pickle


def _make_state_filename(df_path, ssp=None):
    """
    根据 df_path 和 ssp 生成状态文件名，保存在当前代码所在目录。
    """
    # 当前 .py 文件所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    csv_name = os.path.basename(df_path)         # 如 Positive_Negative_balanced_25366.csv
    csv_stem, _ = os.path.splitext(csv_name)     # Positive_Negative_balanced_25366

    if ssp is not None and str(ssp).strip() != "":
        ssp_str = str(ssp).strip().replace(" ", "")
        fname = f"feature_state_{csv_stem}_{ssp_str}.pkl"
    else:
        fname = f"feature_state_{csv_stem}.pkl"

    return os.path.join(base_dir, fname)


def save_feature_state(
    df_path,
    ssp,
    sinkhole_position,
    database_folder_path,
    input_folder_path,
    output_folder_path,
    historical_folder_path,
    future_folder_path,
    future_ssp_folder_path,
    extra_vars=None,
):
    """
    在需求10完成后调用：把关键“全局变量”打包保存成一个 pickle 文件。

    必须参数：
    --------
    df_path : str
        主代码读取的样本 CSV 路径。
    ssp : str
        当前 SSP 情景，比如 'ssp1', 'ssp2', 'ssp3', 'ssp5'。
    sinkhole_position : pandas.DataFrame
        已经完成需求1–10，包含所有特征的主数据表。
    database_folder_path, input_folder_path, output_folder_path,
    historical_folder_path, future_folder_path, future_ssp_folder_path : str
        主程序中用到的关键路径变量。

    extra_vars : dict, optional
        如果你还有其他想一起保存的变量，可以通过字典形式传进来，
        会一起打包进 state。

    返回：
    ----
    state_path : str
        保存的 pickle 文件完整路径。
    """
    state = {
        "df_path": df_path,
        "ssp": ssp,
        "sinkhole_position": sinkhole_position,
        "database_folder_path": database_folder_path,
        "input_folder_path": input_folder_path,
        "output_folder_path": output_folder_path,
        "historical_folder_path": historical_folder_path,
        "future_folder_path": future_folder_path,
        "future_ssp_folder_path": future_ssp_folder_path,
    }

    if extra_vars is not None:
        # 用户自定义的变量也一起塞进去
        state.update(extra_vars)

    state_path = _make_state_filename(df_path, ssp)

    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    print("\n[FeatureState] 已保存特征提取后的全局变量状态：")
    print("  路径:", state_path)
    print("  含变量键:", list(state.keys()))

    return state_path


def load_feature_state(df_path, ssp=None):
    """
    在后续脚本中调用：根据 df_path + ssp 读取之前保存的状态文件，
    返回一个包含所有变量的字典。

    用法示例：
    --------
    state = load_feature_state(df_path, ssp)
    sinkhole_position = state["sinkhole_position"]
    output_folder_path = state["output_folder_path"]
    ...
    """
    state_path = _make_state_filename(df_path, ssp)

    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"[FeatureState] 未找到状态文件: {state_path}\n"
            f"请先在完成需求10后调用 save_feature_state 保存一次。"
        )

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    print("\n[FeatureState] 已读取特征提取后的全局变量状态：")
    print("  路径:", state_path)
    print("  变量键:", list(state.keys()))

    return state
