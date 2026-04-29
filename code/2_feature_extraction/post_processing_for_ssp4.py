import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from feature_state_io import load_feature_state


# ===== ssp4 missing feature column =====
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

# Total 40 columns: 5(LAI) + 5(Precip) + 5(WTD) + 5(HDS) + 5(Tas) + 5(Tasmax) + 5(Tasmin) + 5(Huss)
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
    """Convert the target column to a numerical value, and set it to NaN if it cannot be converted."""
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
    """Align with key_col (default No) and average the specified feature columns of the two data sets (NaN is ignored).

    Return:
      avg_df: index=No, columns=feature_cols
      missing_in_a: df_a missing feature column
      missing_in_b: df_b missing feature column"""
    if key_col not in df_a.columns or key_col not in df_b.columns:
        raise ValueError(f"{label_prefix}The input data is missing the primary key column '{key_col}'.")

    missing_in_a = [c for c in feature_cols if c not in df_a.columns]
    missing_in_b = [c for c in feature_cols if c not in df_b.columns]

    if missing_in_a or missing_in_b:
        print(f"{label_prefix}Warning: A is missing column:{missing_in_a}")
        print(f"{label_prefix}Warning: B is missing column:{missing_in_b}")

    cols_exist = [c for c in feature_cols if (c in df_a.columns and c in df_b.columns)]
    if not cols_exist:
        raise ValueError(f"{label_prefix}The two sets of data do not have any common feature columns to be processed and cannot be generated.")

    da = df_a[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)
    db = df_b[[key_col] + cols_exist].drop_duplicates(subset=[key_col]).set_index(key_col)

    da = _to_numeric_df(da, cols_exist)
    db = _to_numeric_df(db, cols_exist)

    idx = da.index.union(db.index)
    dau = da.reindex(idx)
    dbu = db.reindex(idx)

    arr = np.nanmean(np.stack([dau.values, dbu.values]), axis=0)
    avg = pd.DataFrame(arr, index=idx, columns=cols_exist)

    # Fill in NaN for missing columns to ensure that the output columns are complete and in a fixed order
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
    """Refer to the saving method of each feature subscript:
    - Historical: *_historical_2000_2010_2020.csv (contains hist column)
    - Future: *_future_{ssp}.csv (contains four window columns 2020-2100)"""
    os.makedirs(historical_folder_path, exist_ok=True)
    os.makedirs(future_ssp_folder_path, exist_ok=True)

    base_cols = ["No", "Longitude", "Latitude"]
    for c in base_cols:
        if c not in sinkhole_position.columns:
            raise ValueError(f"{label_prefix}sinkhole_position missing required column: '{c}'")

    def _save_one(prefix_name: str, hist_col: str, future_cols: List[str]):
        # Processing step.
        hist_path = os.path.join(
            historical_folder_path,
            f"{prefix_name}_historical_2000_2010_2020.csv",
        )
        hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
        hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
        sinkhole_position[hist_out_cols].to_csv(hist_path, index=False, encoding="utf-8-sig")

        # Processing step.
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

    print(f"\n{label_prefix}Outputted SSP4 missing feature files:")
    for p in output_paths:
        print("  ", p)



def post_processing_for_ssp4(
    sinkhole_position: pd.DataFrame,
    df_path: str,
    historical_folder_path: str,
    future_folder_path: str,          # Reserved parameters (consistent with the main process), the current implementation is not strongly dependent on
    future_ssp_folder_path: str,
    ssp: str,
) -> pd.DataFrame:
    """Requirements:
      ssp4 is missing the following 40 columns:
      - LAI (5 columns)
      - Precip (5 columns)
      - WTD (5 columns)
      - HDS (5 columns)
      - Tas (5 columns)
      - Tasmax (5 columns)
      - Tasmin (5 columns)
      - Huss (5 columns)

      Use the corresponding columns of ssp3 and ssp5 to align with No and average to generate ssp4.

    Dependencies:
      The feature_state of ssp3 and ssp5 needs to be saved (requires the load_feature_state calling method of 10.52)."""
    _ = future_folder_path  # Processing step.

    ssp_key = str(ssp).lower()
    if ssp_key != "ssp4":
        print(f"[PostSSP4] current ssp={ssp}, no need to perform ssp4 post-processing, skip.")
        return sinkhole_position

    print(
        "\\n[PostSSP4] Start generating missing features for ssp4"
        "(LAI/Precip/WTD/HDS/Tas/Tasmax/Tasmin/Huss: ssp3 vs. ssp5 average)..."
    )

    # 1) Read the global status of ssp3 / ssp5
    try:
        state3 = load_feature_state(df_path, "ssp3")
        state5 = load_feature_state(df_path, "ssp5")
    except Exception as e:
        raise RuntimeError(
            "[PostSSP4] Unable to read global variable status of ssp3/ssp5."
            "ssp3 ssp5 (10.51/10.52)."
        ) from e

    df3 = state3.get("sinkhole_position", None)
    df5 = state5.get("sinkhole_position", None)
    if df3 is None or df5 is None:
        raise RuntimeError("[PostSSP4] ssp3/ssp5 state sinkhole_position.")

    # 2) Calculate the average (aligned by No, ignored by NaN)
    avg_df, _, _ = _average_two_ssps_by_no(
        df_a=df3,
        df_b=df5,
        feature_cols=ALL_FEATURE_COLS,
        key_col="No",
        label_prefix="[PostSSP4]",
    )

    # 3) Write back the current sinkhole_position (maintain the original row order)
    if "No" not in sinkhole_position.columns:
        raise ValueError("[PostSSP4] Current sinkhole_position is missing column 'No' and cannot align writes.")

    sp = sinkhole_position.copy()
    sp_idx = sp.drop_duplicates(subset=["No"]).set_index("No", drop=False)

    avg_aligned = avg_df.reindex(sp_idx.index)

    n_missing_no = int(avg_aligned.isna().all(axis=1).sum())
    if n_missing_no > 0:
        print(f"[PostSSP4] Warning: Yes{n_missing_no}points are missing in ssp3/ssp5 (still all NaN after averaging).")

    for c in ALL_FEATURE_COLS:
        sp_idx[c] = avg_aligned[c].values

    sp_out = sp_idx.reset_index(drop=True)

    # 4) Save the output (refer to the "history/future" two types of CSV of each feature script)
    _save_feature_outputs(
        sinkhole_position=sp_out,
        historical_folder_path=historical_folder_path,
        future_ssp_folder_path=future_ssp_folder_path,
        ssp=ssp,
        label_prefix="[PostSSP4]",
    )

    print("[PostSSP4] ssp4 missing feature generation completed.")
    return sp_out
