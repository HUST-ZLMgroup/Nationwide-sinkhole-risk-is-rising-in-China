from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lingam_pipeline_v1.pre_attribution.attribution_config import OUTPUT_ROOT, SSPS, YEARS


CODE_V2_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_V2_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_V2_DIR))

from mgtwr.gwr_sigmoid_utils import gwr_scores_to_probabilities  # noqa: E402


DEFAULT_PATH_NAME = "Points_China_all_10km"
DEFAULT_POINTS_BASE_DIR = PROJECT_ROOT / "data" / "Extracted_HAVE_future"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "code" / "5_gwr_model_prediction"
DEFAULT_CLASSIFICATION_ARTIFACTS = (
    PROJECT_ROOT / "code" / "3_gwr_model_train" / "national" / "GWR" / "gwr_classification_artifacts.pkl"
)

WINDOW_BY_YEAR = {
    "2040": "2020_2040",
    "2060": "2040_2060",
    "2080": "2060_2080",
    "2100": "2080_2100",
}

STATIC_COLUMNS = {
    "DK": "Distance_to_karst",
    "DB": "Depth_to_Bedrock",
    "DF": "Distance_to_Fault_m",
}

DYNAMIC_COLUMN_TEMPLATES = {
    "HDS": "HDS_{window}",
    "PR": "Precip_{window}",
    "TAS": "Tas_{window}",
    "HUSS": "Huss_{window}",
    "PT": "PopTotal_{window}",
    "UF": "UrbanFrac_{window}",
    "IP": "ImperviousIndex_{window}",
    "LAI": "LAI_{window}",
    "WTD": "WTD_{window}",
}

METADATA_COLUMNS = ["sample_id", "No", "Longitude", "Latitude", "ADCODE99", "NAME_EN_JX"]


def _parse_requested(values: list[str], allowed: tuple[str, ...], label: str) -> list[str]:
    if not values:
        return list(allowed)
    normalized = [str(v).lower() for v in values]
    if "all" in normalized:
        return list(allowed)

    requested = [str(v) for v in values]
    invalid = [v for v in requested if v not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {label}: {invalid}. Allowed values: {list(allowed)}")
    return requested


def _load_transform_metadata(classification_artifacts_path: str | Path) -> dict[str, float]:
    with Path(classification_artifacts_path).open("rb") as f:
        artifacts = pickle.load(f)
    transform_metadata = artifacts.get("transform_metadata")
    if not transform_metadata:
        raise KeyError(
            f"transform_metadata is missing from classification artifacts: {classification_artifacts_path}"
        )
    return transform_metadata


def _point_csv_path(points_base_dir: str | Path, path_name: str, ssp: str) -> Path:
    return Path(points_base_dir) / path_name / f"AllFeatures_{path_name}_{ssp}_cleaned.csv"


def _prediction_pkl_path(prediction_dir: str | Path, path_name: str, ssp: str, year: str) -> Path:
    return Path(prediction_dir) / f"gwr_pre_{path_name}_{ssp}_{year}_results.pkl"


def _rename_map_for_year(year: str) -> dict[str, str]:
    if year not in WINDOW_BY_YEAR:
        raise KeyError(f"Unsupported year for future window mapping: {year}")
    window = WINDOW_BY_YEAR[year]
    rename_map = {source_col: node for node, source_col in STATIC_COLUMNS.items()}
    for node, template in DYNAMIC_COLUMN_TEMPLATES.items():
        rename_map[template.format(window=window)] = node
    return rename_map


def build_future_dataset_for_scenario(
    ssp: str,
    year: str,
    path_name: str = DEFAULT_PATH_NAME,
    points_base_dir: str | Path = DEFAULT_POINTS_BASE_DIR,
    prediction_dir: str | Path = DEFAULT_PREDICTION_DIR,
    classification_artifacts_path: str | Path = DEFAULT_CLASSIFICATION_ARTIFACTS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    point_csv = _point_csv_path(points_base_dir, path_name, ssp)
    prediction_pkl = _prediction_pkl_path(prediction_dir, path_name, ssp, year)
    if not point_csv.exists():
        raise FileNotFoundError(f"Future points CSV not found: {point_csv}")
    if not prediction_pkl.exists():
        raise FileNotFoundError(f"Future GWR prediction PKL not found: {prediction_pkl}")

    source_df = pd.read_csv(point_csv)
    rename_map = _rename_map_for_year(year)
    missing_cols = [col for col in rename_map if col not in source_df.columns]
    if missing_cols:
        raise KeyError(f"Future source CSV is missing required columns for {ssp}_{year}: {missing_cols}")

    with prediction_pkl.open("rb") as f:
        prediction_obj = pickle.load(f)
    if "y_pred_gwr" not in prediction_obj:
        raise KeyError(f"'y_pred_gwr' is missing from prediction PKL: {prediction_pkl}")
    gwr_raw_score = np.asarray(prediction_obj["y_pred_gwr"], dtype=float).reshape(-1)
    if len(source_df) != len(gwr_raw_score):
        raise ValueError(
            "Future points CSV row count does not match GWR raw-score length:\n"
            f"len(source_df)={len(source_df)}, len(gwr_raw_score)={len(gwr_raw_score)}"
        )

    transform_metadata = _load_transform_metadata(classification_artifacts_path)
    risk_probability, _ = gwr_scores_to_probabilities(
        gwr_raw_score,
        transform_metadata=transform_metadata,
        return_metadata=True,
    )
    risk_probability = np.asarray(risk_probability, dtype=float).reshape(-1)

    keep_cols = ["No", "Longitude", "Latitude", "ADCODE99", "NAME_EN_JX"] + list(rename_map.keys())
    df = source_df.loc[:, keep_cols].copy().rename(columns=rename_map)
    df["sample_id"] = pd.to_numeric(df["No"], errors="coerce")
    df["scenario_id"] = f"{ssp}_{year}"
    df["ssp"] = ssp
    df["year"] = year
    df["gwr_raw_score"] = gwr_raw_score
    df["risk_probability"] = risk_probability
    df["RISK"] = df["risk_probability"]

    ordered_cols = (
        METADATA_COLUMNS
        + ["scenario_id", "ssp", "year"]
        + ["DK", "DB", "DF", "HDS", "PR", "TAS", "HUSS", "PT", "UF", "IP", "LAI", "WTD"]
        + ["gwr_raw_score", "risk_probability", "RISK"]
    )
    df = df.loc[:, ordered_cols]

    numeric_cols = [
        "sample_id",
        "Longitude",
        "Latitude",
        "DK",
        "DB",
        "DF",
        "HDS",
        "PR",
        "TAS",
        "HUSS",
        "PT",
        "UF",
        "IP",
        "LAI",
        "WTD",
        "gwr_raw_score",
        "risk_probability",
        "RISK",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=0).reset_index(drop=True)
    dropped_rows = before_drop - len(df)

    metadata = {
        "scenario_id": f"{ssp}_{year}",
        "ssp": ssp,
        "year": year,
        "path_name": path_name,
        "points_csv": str(point_csv.resolve()),
        "prediction_pkl": str(prediction_pkl.resolve()),
        "classification_artifacts_path": str(Path(classification_artifacts_path).resolve()),
        "transform_metadata": transform_metadata,
        "n_rows_original": int(before_drop),
        "n_rows_output": int(len(df)),
        "n_rows_dropped_for_na": int(dropped_rows),
        "gwr_raw_score_stats": {
            "min": float(np.nanmin(df["gwr_raw_score"])),
            "max": float(np.nanmax(df["gwr_raw_score"])),
            "mean": float(np.nanmean(df["gwr_raw_score"])),
        },
        "risk_probability_stats": {
            "min": float(np.nanmin(df["risk_probability"])),
            "max": float(np.nanmax(df["risk_probability"])),
            "mean": float(np.nanmean(df["risk_probability"])),
        },
    }
    return df, metadata


def write_future_dataset_for_scenario(
    ssp: str,
    year: str,
    path_name: str = DEFAULT_PATH_NAME,
    points_base_dir: str | Path = DEFAULT_POINTS_BASE_DIR,
    prediction_dir: str | Path = DEFAULT_PREDICTION_DIR,
    classification_artifacts_path: str | Path = DEFAULT_CLASSIFICATION_ARTIFACTS,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    out_root = Path(output_root) if output_root is not None else OUTPUT_ROOT / "future"
    out_dir = out_root / ssp / year
    out_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = build_future_dataset_for_scenario(
        ssp=ssp,
        year=year,
        path_name=path_name,
        points_base_dir=points_base_dir,
        prediction_dir=prediction_dir,
        classification_artifacts_path=classification_artifacts_path,
    )
    csv_path = out_dir / "future_samples_for_lingam.csv"
    meta_path = out_dir / "future_samples_for_lingam_metadata.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build future-stage LiNGAM sample tables from future drivers and saved GWR raw predictions.")
    parser.add_argument(
        "--ssp",
        nargs="+",
        default=["all"],
        help="SSP ids to build. Use one or more of ssp1..ssp5, or 'all'.",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        default=["all"],
        help="Future years to build. Use one or more of 2040/2060/2080/2100, or 'all'.",
    )
    parser.add_argument(
        "--path-name",
        type=str,
        default=DEFAULT_PATH_NAME,
        help="Dataset path name used in both future driver CSVs and saved GWR prediction PKLs.",
    )
    parser.add_argument(
        "--points-base-dir",
        type=Path,
        default=DEFAULT_POINTS_BASE_DIR,
        help="Base directory that contains the extracted future point CSV folders.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help="Directory that contains saved future GWR prediction PKLs.",
    )
    parser.add_argument(
        "--classification-artifacts-path",
        type=Path,
        default=DEFAULT_CLASSIFICATION_ARTIFACTS,
        help="Training-stage GWR classification artifacts PKL with transform_metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT / "future",
        help="Root directory for future-stage LiNGAM sample tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_ssps = _parse_requested(args.ssp, SSPS, "ssp")
    requested_years = _parse_requested(args.year, YEARS, "year")

    for ssp in requested_ssps:
        for year in requested_years:
            paths = write_future_dataset_for_scenario(
                ssp=ssp,
                year=year,
                path_name=args.path_name,
                points_base_dir=args.points_base_dir,
                prediction_dir=args.prediction_dir,
                classification_artifacts_path=args.classification_artifacts_path,
                output_root=args.output_root,
            )
            print(f"[OK] {ssp}_{year} future_samples_for_lingam.csv -> {paths['csv']}")
            print(f"[OK] {ssp}_{year} future_samples_for_lingam_metadata.json -> {paths['metadata']}")


if __name__ == "__main__":
    main()
