import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from feature_state_io import load_feature_state


# ===== 需求指定的 10 个特征列（5个WTD + 5个HDS）=====
WTD_COLS: List[str] = [
    "WTD_hist_2000_2010_2020",
    "WTD_2020_2040",
    "WTD_2040_2060",
    "WTD_2060_2080",
    "WTD_2080_2100",
]

HDS_COLS: List[str] = [
    "HDS_hist_2000_2010_2020",
    "HDS_2020_2040",
    "HDS_2040_2060",
    "HDS_2060_2080",
    "HDS_2080_2100",
]

ALL_FEATURE_COLS: List[str] = WTD_COLS + HDS_COLS


def _to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """把目标列转为数值，无法转换的置为 NaN。"""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _average_two_ssps_by_no(
    df_ssp1: pd.DataFrame,
    df_ssp3: pd.DataFrame,
    feature_cols: List[str],
    key_col: str = "No",
    label_prefix: str = "[PostSSP2]",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    以 key_col（默认 No）对齐，将 SSP1 与 SSP3 的指定特征列做平均（NaN 忽略）。

    返回：
      avg_df: index=No, columns=feature_cols
      missing_in_ssp1: SSP1 缺少的特征列
      missing_in_ssp3: SSP3 缺少的特征列
    """
    if key_col not in df_ssp1.columns or key_col not in df_ssp3.columns:
        raise ValueError(f"{label_prefix} ssp1/ssp3 的 sinkhole_position 缺少主键列 '{key_col}'。")

    missing_in_ssp1 = [c for c in feature_cols if c not in df_ssp1.columns]
    missing_in_ssp3 = [c for c in feature_cols if c not in df_ssp3.columns]

    if missing_in_ssp1 or missing_in_ssp3:
        print(f"{label_prefix} 警告：ssp1 缺少列: {missing_in_ssp1}")
        print(f"{label_prefix} 警告：ssp3 缺少列: {missing_in_ssp3}")

    # 只对两边都存在的列做平均
    cols_exist = [c for c in feature_cols if (c in df_ssp1.columns and c in df_ssp3.columns)]
    if not cols_exist:
        raise ValueError(f"{label_prefix} ssp1 与 ssp3 没有任何共同的待处理特征列，无法生成 ssp2。")

    d1 = df_ssp1[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)
    d3 = df_ssp3[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)

    d1 = _to_numeric_df(d1, cols_exist)
    d3 = _to_numeric_df(d3, cols_exist)

    # union index：允许某些点位只在一边出现（最终会得到 NaN 或单边值）
    idx = d1.index.union(d3.index)
    d1u = d1.reindex(idx)
    d3u = d3.reindex(idx)

    # NaN 忽略平均：若一边为 NaN、一边有值 -> 取有值；两边都 NaN -> NaN
    arr = np.nanmean(np.stack([d1u.values, d3u.values]), axis=0)
    avg = pd.DataFrame(arr, index=idx, columns=cols_exist)

    # 对缺失列补齐 NaN，确保输出列完整
    for c in feature_cols:
        if c not in avg.columns:
            avg[c] = np.nan

    avg = avg[feature_cols]  # 保持顺序
    return avg, missing_in_ssp1, missing_in_ssp3


def _save_groundwater_like_outputs(
    sinkhole_position: pd.DataFrame,
    historical_folder_path: str,
    future_ssp_folder_path: str,
    ssp: str,
    label_prefix: str = "[PostSSP2]",
) -> None:
    """
    按 groundwater_wtd.py / groundwater_hds.py 的输出命名规则保存，
    以便后续 aggregate_features 能直接汇总。
    """
    os.makedirs(historical_folder_path, exist_ok=True)
    os.makedirs(future_ssp_folder_path, exist_ok=True)

    base_cols = ["No", "Longitude", "Latitude"]
    for c in base_cols:
        if c not in sinkhole_position.columns:
            raise ValueError(f"{label_prefix} sinkhole_position 缺少必要列: '{c}'")

    # ===== WTD 输出 =====
    wtd_hist_cols = ["No", "Longitude", "Latitude", "WTD_hist_2000_2010_2020"]
    wtd_hist_cols = [c for c in wtd_hist_cols if c in sinkhole_position.columns]
    wtd_hist_path = os.path.join(historical_folder_path, "GroundwaterWTD_historical_2000_2010_2020.csv")
    sinkhole_position[wtd_hist_cols].to_csv(wtd_hist_path, index=False, encoding="utf-8-sig")

    wtd_future_cols = ["No", "Longitude", "Latitude"] + [c for c in WTD_COLS if c != "WTD_hist_2000_2010_2020"]
    wtd_future_cols = [c for c in wtd_future_cols if c in sinkhole_position.columns]
    wtd_future_path = os.path.join(future_ssp_folder_path, f"GroundwaterWTD_future_{ssp}.csv")
    sinkhole_position[wtd_future_cols].to_csv(wtd_future_path, index=False, encoding="utf-8-sig")

    # ===== HDS 输出 =====
    hds_hist_cols = ["No", "Longitude", "Latitude", "HDS_hist_2000_2010_2020"]
    hds_hist_cols = [c for c in hds_hist_cols if c in sinkhole_position.columns]
    hds_hist_path = os.path.join(historical_folder_path, "GroundwaterHDS_historical_2000_2010_2020.csv")
    sinkhole_position[hds_hist_cols].to_csv(hds_hist_path, index=False, encoding="utf-8-sig")

    hds_future_cols = ["No", "Longitude", "Latitude"] + [c for c in HDS_COLS if c != "HDS_hist_2000_2010_2020"]
    hds_future_cols = [c for c in hds_future_cols if c in sinkhole_position.columns]
    hds_future_path = os.path.join(future_ssp_folder_path, f"GroundwaterHDS_future_{ssp}.csv")
    sinkhole_position[hds_future_cols].to_csv(hds_future_path, index=False, encoding="utf-8-sig")

    print(f"\n{label_prefix} 已输出 SSP2 地下水特征：")
    print("  ", wtd_hist_path)
    print("  ", wtd_future_path)
    print("  ", hds_hist_path)
    print("  ", hds_future_path)


def post_processing_for_ssp2(
    sinkhole_position: pd.DataFrame,
    df_path: str,
    historical_folder_path: str,
    future_folder_path: str,          # 预留参数（与主流程一致），当前实现不强依赖
    future_ssp_folder_path: str,
    ssp: str,
) -> pd.DataFrame:
    """
    需求：
      由于地下水 WTD/HDS 没有 ssp2 原始数据，通过 ssp1 与 ssp3 做平均生成 ssp2。

    实现要点：
    - ssp1/ssp3 数据通过 load_feature_state(df_path, sspX) 读取（与需求10.52一致）。
    - 生成的 10 个特征列写回 sinkhole_position，并输出到 historical / future/ssp2 文件夹。
    """
    ssp_key = str(ssp).lower()
    if ssp_key != "ssp2":
        print(f"[PostSSP2] 当前 ssp={ssp}，无需执行 ssp2 后处理，跳过。")
        return sinkhole_position

    print("\n[PostSSP2] 开始为 ssp2 生成地下水 WTD/HDS（ssp1 与 ssp3 平均）...")

    # 1) 读取 ssp1 / ssp3 的全局状态（其中应包含已提取的 WTD/HDS 列）
    try:
        state1 = load_feature_state(df_path, "ssp1")
        state3 = load_feature_state(df_path, "ssp3")
    except Exception as e:
        raise RuntimeError(
            "[PostSSP2] 无法读取 ssp1/ssp3 的全局变量状态。"
            "请先分别运行并保存 ssp1 与 ssp3 的特征状态（需求10.51）。"
        ) from e

    df1 = state1.get("sinkhole_position", None)
    df3 = state3.get("sinkhole_position", None)
    if df1 is None or df3 is None:
        raise RuntimeError("[PostSSP2] ssp1/ssp3 的 state 中未找到 sinkhole_position。")

    # 2) 计算平均值（按 No 对齐，NaN 忽略）
    avg_df, _, _ = _average_two_ssps_by_no(
        df_ssp1=df1,
        df_ssp3=df3,
        feature_cols=ALL_FEATURE_COLS,
        key_col="No",
        label_prefix="[PostSSP2]",
    )

    # 3) 写回当前 sinkhole_position（保持原行顺序）
    if "No" not in sinkhole_position.columns:
        raise ValueError("[PostSSP2] 当前 sinkhole_position 缺少 'No' 列，无法对齐写入。")

    sp = sinkhole_position.copy()
    sp_idx = sp.drop_duplicates(subset=["No"]).set_index("No", drop=False)

    avg_aligned = avg_df.reindex(sp_idx.index)

    # 对齐统计：如果某些点在 ssp1/ssp3 都缺失，会是全 NaN
    n_missing_no = int(avg_aligned.isna().all(axis=1).sum())
    if n_missing_no > 0:
        print(f"[PostSSP2] 警告：有 {n_missing_no} 个点位在 ssp1/ssp3 中都缺失（平均后仍为全 NaN）。")

    for c in ALL_FEATURE_COLS:
        sp_idx[c] = avg_aligned[c].values

    sp_out = sp_idx.reset_index(drop=True)

    # 4) 保存输出（仿照 groundwater_wtd/groundwater_hds 的文件命名）
    _save_groundwater_like_outputs(
        sinkhole_position=sp_out,
        historical_folder_path=historical_folder_path,
        future_ssp_folder_path=future_ssp_folder_path,
        ssp=ssp,
        label_prefix="[PostSSP2]",
    )

    print("[PostSSP2] ssp2 地下水特征生成完成。")
    return sp_out
