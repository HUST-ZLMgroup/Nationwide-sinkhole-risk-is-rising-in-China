import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from feature_state_io import load_feature_state


# ===== 10 feature columns specified by requirements (5 WTD + 5 HDS) =====
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
    """Convert the target column to a numerical value, and set it to NaN if it cannot be converted."""
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
    """key_col( No), SSP1 SSP3 (NaN ). : avg_df: index=No, columns=feature_cols missing_in_ssp1: SSP1 missing_in_ssp3: SSP3"""
    if key_col not in df_ssp1.columns or key_col not in df_ssp3.columns:
        raise ValueError(f"{label_prefix}sinkhole_position of ssp1/ssp3 is missing primary key column '{key_col}'.")

    missing_in_ssp1 = [c for c in feature_cols if c not in df_ssp1.columns]
    missing_in_ssp3 = [c for c in feature_cols if c not in df_ssp3.columns]

    if missing_in_ssp1 or missing_in_ssp3:
        print(f"{label_prefix}Warning: ssp1 missing column:{missing_in_ssp1}")
        print(f"{label_prefix}Warning: ssp3 missing column:{missing_in_ssp3}")

    # Only average the columns that exist on both sides
    cols_exist = [c for c in feature_cols if (c in df_ssp1.columns and c in df_ssp3.columns)]
    if not cols_exist:
        raise ValueError(f"{label_prefix}ssp1 and ssp3 do not have any common feature columns to be processed, so ssp2 cannot be generated.")

    d1 = df_ssp1[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)
    d3 = df_ssp3[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)

    d1 = _to_numeric_df(d1, cols_exist)
    d3 = _to_numeric_df(d3, cols_exist)

    # union index: Allow certain points to appear on only one side (you will eventually get NaN or single-sided value)
    idx = d1.index.union(d3.index)
    d1u = d1.reindex(idx)
    d3u = d3.reindex(idx)

    # NaN ignores the average: if one side is NaN and one side has a value -> take a value; both sides are NaN -> NaN
    arr = np.nanmean(np.stack([d1u.values, d3u.values]), axis=0)
    avg = pd.DataFrame(arr, index=idx, columns=cols_exist)

    # Fill in NaN for missing columns to ensure that the output columns are complete
    for c in feature_cols:
        if c not in avg.columns:
            avg[c] = np.nan

    avg = avg[feature_cols]  # Keep order
    return avg, missing_in_ssp1, missing_in_ssp3


def _save_groundwater_like_outputs(
    sinkhole_position: pd.DataFrame,
    historical_folder_path: str,
    future_ssp_folder_path: str,
    ssp: str,
    label_prefix: str = "[PostSSP2]",
) -> None:
    """Save according to the output naming rules of groundwater_wtd.py / groundwater_hds.py,
    So that subsequent aggregate_features can be summarized directly."""
    os.makedirs(historical_folder_path, exist_ok=True)
    os.makedirs(future_ssp_folder_path, exist_ok=True)

    base_cols = ["No", "Longitude", "Latitude"]
    for c in base_cols:
        if c not in sinkhole_position.columns:
            raise ValueError(f"{label_prefix}sinkhole_position missing required column: '{c}'")

    # ===== WTD =====
    wtd_hist_cols = ["No", "Longitude", "Latitude", "WTD_hist_2000_2010_2020"]
    wtd_hist_cols = [c for c in wtd_hist_cols if c in sinkhole_position.columns]
    wtd_hist_path = os.path.join(historical_folder_path, "GroundwaterWTD_historical_2000_2010_2020.csv")
    sinkhole_position[wtd_hist_cols].to_csv(wtd_hist_path, index=False, encoding="utf-8-sig")

    wtd_future_cols = ["No", "Longitude", "Latitude"] + [c for c in WTD_COLS if c != "WTD_hist_2000_2010_2020"]
    wtd_future_cols = [c for c in wtd_future_cols if c in sinkhole_position.columns]
    wtd_future_path = os.path.join(future_ssp_folder_path, f"GroundwaterWTD_future_{ssp}.csv")
    sinkhole_position[wtd_future_cols].to_csv(wtd_future_path, index=False, encoding="utf-8-sig")

    # ===== HDS output =====
    hds_hist_cols = ["No", "Longitude", "Latitude", "HDS_hist_2000_2010_2020"]
    hds_hist_cols = [c for c in hds_hist_cols if c in sinkhole_position.columns]
    hds_hist_path = os.path.join(historical_folder_path, "GroundwaterHDS_historical_2000_2010_2020.csv")
    sinkhole_position[hds_hist_cols].to_csv(hds_hist_path, index=False, encoding="utf-8-sig")

    hds_future_cols = ["No", "Longitude", "Latitude"] + [c for c in HDS_COLS if c != "HDS_hist_2000_2010_2020"]
    hds_future_cols = [c for c in hds_future_cols if c in sinkhole_position.columns]
    hds_future_path = os.path.join(future_ssp_folder_path, f"GroundwaterHDS_future_{ssp}.csv")
    sinkhole_position[hds_future_cols].to_csv(hds_future_path, index=False, encoding="utf-8-sig")

    print(f"\n{label_prefix}SSP2 :")
    print("  ", wtd_hist_path)
    print("  ", wtd_future_path)
    print("  ", hds_hist_path)
    print("  ", hds_future_path)


def post_processing_for_ssp2(
    sinkhole_position: pd.DataFrame,
    df_path: str,
    historical_folder_path: str,
    future_folder_path: str,          # Reserved parameters (consistent with the main process), the current implementation is not strongly dependent on
    future_ssp_folder_path: str,
    ssp: str,
) -> pd.DataFrame:
    """Requirements:
      Since groundwater WTD/HDS does not have ssp2 original data, ssp2 is generated by averaging ssp1 and ssp3.

    Implementation points:
    - ssp1/ssp3 data is read through load_feature_state(df_path, sspX) (consistent with requirement 10.52).
    - The generated 10 feature columns are written back to sinkhole_position and output to the historical/future/ssp2 folder."""
    ssp_key = str(ssp).lower()
    if ssp_key != "ssp2":
        print(f"[PostSSP2] ssp={ssp}, no need to perform ssp2 post-processing, skip.")
        return sinkhole_position

    print("\\n[PostSSP2] Start generating groundwater WTD/HDS for ssp2 (average of ssp1 and ssp3)...")

    # 1) ssp1 / ssp3 ( WTD/HDS )
    try:
        state1 = load_feature_state(df_path, "ssp1")
        state3 = load_feature_state(df_path, "ssp3")
    except Exception as e:
        raise RuntimeError(
            "[PostSSP2] Unable to read global variable status of ssp1/ssp3."
            "ssp1 ssp3 (10.51)."
        ) from e

    df1 = state1.get("sinkhole_position", None)
    df3 = state3.get("sinkhole_position", None)
    if df1 is None or df3 is None:
        raise RuntimeError("[PostSSP2] sinkhole_position not found in state of ssp1/ssp3.")

    # 2) Calculate the average (aligned by No, ignored by NaN)
    avg_df, _, _ = _average_two_ssps_by_no(
        df_ssp1=df1,
        df_ssp3=df3,
        feature_cols=ALL_FEATURE_COLS,
        key_col="No",
        label_prefix="[PostSSP2]",
    )

    # 3) Write back the current sinkhole_position (maintain the original row order)
    if "No" not in sinkhole_position.columns:
        raise ValueError("[PostSSP2] Current sinkhole_position is missing column 'No' and cannot align writes.")

    sp = sinkhole_position.copy()
    sp_idx = sp.drop_duplicates(subset=["No"]).set_index("No", drop=False)

    avg_aligned = avg_df.reindex(sp_idx.index)

    # Alignment statistics: If some points are missing in ssp1/ssp3, it will be all NaN
    n_missing_no = int(avg_aligned.isna().all(axis=1).sum())
    if n_missing_no > 0:
        print(f"[PostSSP2] Warning: Yes{n_missing_no}points are missing in ssp1/ssp3 (still all NaN after averaging).")

    for c in ALL_FEATURE_COLS:
        sp_idx[c] = avg_aligned[c].values

    sp_out = sp_idx.reset_index(drop=True)

    # 4) Save the output (named after the file of groundwater_wtd/groundwater_hds)
    _save_groundwater_like_outputs(
        sinkhole_position=sp_out,
        historical_folder_path=historical_folder_path,
        future_ssp_folder_path=future_ssp_folder_path,
        ssp=ssp,
        label_prefix="[PostSSP2]",
    )

    print("[PostSSP2] ssp2 groundwater feature generation completed.")
    return sp_out
