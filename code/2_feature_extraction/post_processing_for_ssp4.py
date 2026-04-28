import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from feature_state_io import load_feature_state


# ===== ssp4 缺失特征列 =====
# 10_leaf_area_index
LAI_COLS: List[str] = [
    "LAI_hist_2000_2010_2020",
    "LAI_2020_2040",
    "LAI_2040_2060",
    "LAI_2060_2080",
    "LAI_2080_2100",
]

# 17_Precipitation
PRECIP_COLS: List[str] = [
    "Precip_hist_2000_2010_2020",
    "Precip_2020_2040",
    "Precip_2040_2060",
    "Precip_2060_2080",
    "Precip_2080_2100",
]

# 18_groundwater
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

# 20_tas
TAS_COLS: List[str] = [
    "Tas_hist_2000_2010_2020",
    "Tas_2020_2040",
    "Tas_2040_2060",
    "Tas_2060_2080",
    "Tas_2080_2100",
]

# 21_tasmax
TASMAX_COLS: List[str] = [
    "Tasmax_hist_2000_2010_2020",
    "Tasmax_2020_2040",
    "Tasmax_2040_2060",
    "Tasmax_2060_2080",
    "Tasmax_2080_2100",
]

# 22_tasmin
TASMIN_COLS: List[str] = [
    "Tasmin_hist_2000_2010_2020",
    "Tasmin_2020_2040",
    "Tasmin_2040_2060",
    "Tasmin_2060_2080",
    "Tasmin_2080_2100",
]

# 23_huss
HUSS_COLS: List[str] = [
    "Huss_hist_2000_2010_2020",
    "Huss_2020_2040",
    "Huss_2040_2060",
    "Huss_2060_2080",
    "Huss_2080_2100",
]

# 共 40 列：5(LAI) + 5(Precip) + 5(WTD) + 5(HDS) + 5(Tas) + 5(Tasmax) + 5(Tasmin) + 5(Huss)
ALL_FEATURE_COLS: List[str] = (
    LAI_COLS
    + PRECIP_COLS
    + WTD_COLS
    + HDS_COLS
    + TAS_COLS
    + TASMAX_COLS
    + TASMIN_COLS
    + HUSS_COLS
)


def _to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """把目标列转为数值，无法转换的置为 NaN。"""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _average_two_ssps_by_no(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    feature_cols: List[str],
    key_col: str = "No",
    label_prefix: str = "[PostSSP4]",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    以 key_col（默认 No）对齐，将两份数据的指定特征列做平均（NaN 忽略）。

    返回：
      avg_df: index=No, columns=feature_cols
      missing_in_a: df_a 缺少的特征列
      missing_in_b: df_b 缺少的特征列
    """
    if key_col not in df_a.columns or key_col not in df_b.columns:
        raise ValueError(f"{label_prefix} 输入数据缺少主键列 '{key_col}'。")

    missing_in_a = [c for c in feature_cols if c not in df_a.columns]
    missing_in_b = [c for c in feature_cols if c not in df_b.columns]

    if missing_in_a or missing_in_b:
        print(f"{label_prefix} 警告：A 缺少列: {missing_in_a}")
        print(f"{label_prefix} 警告：B 缺少列: {missing_in_b}")

    cols_exist = [c for c in feature_cols if (c in df_a.columns and c in df_b.columns)]
    if not cols_exist:
        raise ValueError(f"{label_prefix} 两份数据没有任何共同的待处理特征列，无法生成。")

    da = df_a[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)
    db = df_b[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)

    da = _to_numeric_df(da, cols_exist)
    db = _to_numeric_df(db, cols_exist)

    idx = da.index.union(db.index)
    dau = da.reindex(idx)
    dbu = db.reindex(idx)

    arr = np.nanmean(np.stack([dau.values, dbu.values]), axis=0)
    avg = pd.DataFrame(arr, index=idx, columns=cols_exist)

    # 对缺失列补齐 NaN，确保输出列完整且顺序固定
    for c in feature_cols:
        if c not in avg.columns:
            avg[c] = np.nan
    avg = avg[feature_cols]
    return avg, missing_in_a, missing_in_b


def _save_feature_outputs(
    sinkhole_position: pd.DataFrame,
    historical_folder_path: str,
    future_ssp_folder_path: str,
    ssp: str,
    label_prefix: str = "[PostSSP4]",
) -> None:
    """
    参考各特征子脚本的保存方式：
    - 历史：*_historical_2000_2010_2020.csv（包含 hist 列）
    - 未来：*_future_{ssp}.csv（包含 2020-2100 四个窗口列）
    """
    os.makedirs(historical_folder_path, exist_ok=True)
    os.makedirs(future_ssp_folder_path, exist_ok=True)

    base_cols = ["No", "Longitude", "Latitude"]
    for c in base_cols:
        if c not in sinkhole_position.columns:
            raise ValueError(f"{label_prefix} sinkhole_position 缺少必要列: '{c}'")

    def _save_one(prefix_name: str, hist_col: str, future_cols: List[str]):
        # 历史
        hist_path = os.path.join(
            historical_folder_path,
            f"{prefix_name}_historical_2000_2010_2020.csv",
        )
        hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
        hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
        sinkhole_position[hist_out_cols].to_csv(hist_path, index=False, encoding="utf-8-sig")

        # 未来
        fut_path = os.path.join(
            future_ssp_folder_path,
            f"{prefix_name}_future_{ssp}.csv",
        )
        fut_out_cols = ["No", "Longitude", "Latitude"] + future_cols
        fut_out_cols = [c for c in fut_out_cols if c in sinkhole_position.columns]
        sinkhole_position[fut_out_cols].to_csv(fut_path, index=False, encoding="utf-8-sig")

        return hist_path, fut_path

    output_paths = []

    # 10 LAI
    output_paths.extend(
        _save_one(
            prefix_name="LeafAreaIndex",
            hist_col="LAI_hist_2000_2010_2020",
            future_cols=[c for c in LAI_COLS if c != "LAI_hist_2000_2010_2020"],
        )
    )

    # 17 Precipitation
    output_paths.extend(
        _save_one(
            prefix_name="PrecipitationAmount",
            hist_col="Precip_hist_2000_2010_2020",
            future_cols=[c for c in PRECIP_COLS if c != "Precip_hist_2000_2010_2020"],
        )
    )

    # 18 GroundwaterWTD
    output_paths.extend(
        _save_one(
            prefix_name="GroundwaterWTD",
            hist_col="WTD_hist_2000_2010_2020",
            future_cols=[c for c in WTD_COLS if c != "WTD_hist_2000_2010_2020"],
        )
    )

    # 18 GroundwaterHDS
    output_paths.extend(
        _save_one(
            prefix_name="GroundwaterHDS",
            hist_col="HDS_hist_2000_2010_2020",
            future_cols=[c for c in HDS_COLS if c != "HDS_hist_2000_2010_2020"],
        )
    )

    # 20 Tas
    output_paths.extend(
        _save_one(
            prefix_name="Tas",
            hist_col="Tas_hist_2000_2010_2020",
            future_cols=[c for c in TAS_COLS if c != "Tas_hist_2000_2010_2020"],
        )
    )

    # 21 Tasmax
    output_paths.extend(
        _save_one(
            prefix_name="Tasmax",
            hist_col="Tasmax_hist_2000_2010_2020",
            future_cols=[c for c in TASMAX_COLS if c != "Tasmax_hist_2000_2010_2020"],
        )
    )

    # 22 Tasmin
    output_paths.extend(
        _save_one(
            prefix_name="Tasmin",
            hist_col="Tasmin_hist_2000_2010_2020",
            future_cols=[c for c in TASMIN_COLS if c != "Tasmin_hist_2000_2010_2020"],
        )
    )

    # 23 Huss
    output_paths.extend(
        _save_one(
            prefix_name="Huss",
            hist_col="Huss_hist_2000_2010_2020",
            future_cols=[c for c in HUSS_COLS if c != "Huss_hist_2000_2010_2020"],
        )
    )

    print(f"\n{label_prefix} 已输出 SSP4 缺失特征文件：")
    for p in output_paths:
        print("  ", p)



def post_processing_for_ssp4(
    sinkhole_position: pd.DataFrame,
    df_path: str,
    historical_folder_path: str,
    future_folder_path: str,          # 预留参数（与主流程一致），当前实现不强依赖
    future_ssp_folder_path: str,
    ssp: str,
) -> pd.DataFrame:
    """
    需求：
      ssp4 缺失以下 40 列：
      - LAI（5列）
      - Precip（5列）
      - WTD（5列）
      - HDS（5列）
      - Tas（5列）
      - Tasmax（5列）
      - Tasmin（5列）
      - Huss（5列）

      用 ssp3 与 ssp5 的对应列按 No 对齐做平均生成 ssp4。

    依赖：
      需要已保存 ssp3 与 ssp5 的 feature_state（需求10.52 的 load_feature_state 调用方式）。
    """
    _ = future_folder_path  # 保留参数以兼容主流程

    ssp_key = str(ssp).lower()
    if ssp_key != "ssp4":
        print(f"[PostSSP4] 当前 ssp={ssp}，无需执行 ssp4 后处理，跳过。")
        return sinkhole_position

    print(
        "\n[PostSSP4] 开始为 ssp4 生成缺失特征"
        "（LAI/Precip/WTD/HDS/Tas/Tasmax/Tasmin/Huss：ssp3 与 ssp5 平均）..."
    )

    # 1) 读取 ssp3 / ssp5 的全局状态
    try:
        state3 = load_feature_state(df_path, "ssp3")
        state5 = load_feature_state(df_path, "ssp5")
    except Exception as e:
        raise RuntimeError(
            "[PostSSP4] 无法读取 ssp3/ssp5 的全局变量状态。"
            "请先分别运行并保存 ssp3 与 ssp5 的特征状态（需求10.51/10.52）。"
        ) from e

    df3 = state3.get("sinkhole_position", None)
    df5 = state5.get("sinkhole_position", None)
    if df3 is None or df5 is None:
        raise RuntimeError("[PostSSP4] ssp3/ssp5 的 state 中未找到 sinkhole_position。")

    # 2) 计算平均值（按 No 对齐，NaN 忽略）
    avg_df, _, _ = _average_two_ssps_by_no(
        df_a=df3,
        df_b=df5,
        feature_cols=ALL_FEATURE_COLS,
        key_col="No",
        label_prefix="[PostSSP4]",
    )

    # 3) 写回当前 sinkhole_position（保持原行顺序）
    if "No" not in sinkhole_position.columns:
        raise ValueError("[PostSSP4] 当前 sinkhole_position 缺少 'No' 列，无法对齐写入。")

    sp = sinkhole_position.copy()
    sp_idx = sp.drop_duplicates(subset=["No"]).set_index("No", drop=False)

    avg_aligned = avg_df.reindex(sp_idx.index)

    n_missing_no = int(avg_aligned.isna().all(axis=1).sum())
    if n_missing_no > 0:
        print(f"[PostSSP4] 警告：有 {n_missing_no} 个点位在 ssp3/ssp5 中都缺失（平均后仍为全 NaN）。")

    for c in ALL_FEATURE_COLS:
        sp_idx[c] = avg_aligned[c].values

    sp_out = sp_idx.reset_index(drop=True)

    # 4) 保存输出（参考各特征脚本的“历史/未来”两类 CSV）
    _save_feature_outputs(
        sinkhole_position=sp_out,
        historical_folder_path=historical_folder_path,
        future_ssp_folder_path=future_ssp_folder_path,
        ssp=ssp,
        label_prefix="[PostSSP4]",
    )

    print("[PostSSP4] ssp4 缺失特征生成完成。")
    return sp_out
