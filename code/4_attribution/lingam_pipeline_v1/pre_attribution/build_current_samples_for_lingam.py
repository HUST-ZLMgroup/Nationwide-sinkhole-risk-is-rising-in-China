from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lingam_pipeline_v1.pre_attribution.attribution_config import OUTPUT_ROOT
from lingam_pipeline_v1.pre_attribution.variable_schema import build_variable_dictionary, node_order


DEFAULT_INPUT_CSV = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "Extracted_HAVE_future"
    / "Positive_Negative_balanced"
    / "AllFeatures_Positive_Negative_balanced_25366_ssp1_cleaned.csv"
)

METADATA_COLUMNS = ["sample_id", "No", "Longitude", "Latitude", "ADCODE99", "NAME_EN_JX", "disaster_observed"]


def build_current_dataset(input_csv: str | Path) -> pd.DataFrame:
    dictionary = build_variable_dictionary()
    source_df = pd.read_csv(input_csv)

    rename_map = {}
    required_columns = []
    for row in dictionary.itertuples(index=False):
        if row.node == "RISK":
            continue
        col = row.example_current_column
        if col not in source_df.columns:
            raise KeyError(f"Missing required current feature column: {col}")
        rename_map[col] = row.node
        required_columns.append(col)

    if "Disaster" not in source_df.columns:
        raise KeyError("Missing required target column: Disaster")

    keep_cols = ["No", "Longitude", "Latitude", "ADCODE99", "NAME_EN_JX", "Disaster"] + required_columns
    df = source_df.loc[:, keep_cols].copy()
    df = df.rename(columns=rename_map)
    df["sample_id"] = df["No"].astype("int64")
    df["disaster_observed"] = pd.to_numeric(df["Disaster"], errors="coerce")
    df["RISK"] = df["disaster_observed"]
    df = df.drop(columns=["Disaster"])

    ordered_cols = METADATA_COLUMNS + node_order()
    df = df.loc[:, ordered_cols]
    numeric_cols = ["Longitude", "Latitude", "disaster_observed"] + node_order()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=0)
    return df.reset_index(drop=True)


def write_current_dataset(input_csv: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / "current"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_current_dataset(input_csv)
    csv_path = out_dir / "current_samples_for_lingam.csv"
    meta_path = out_dir / "current_samples_for_lingam_metadata.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metadata = {
        "input_csv": str(Path(input_csv).resolve()),
        "n_samples": int(len(df)),
        "columns": df.columns.tolist(),
        "target_mode": "observed_disaster_mapped_to_RISK",
        "node_columns": node_order(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the current-stage LiNGAM sample table.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Source CSV for current-stage attribution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "current",
        help="Output directory for current-stage LiNGAM samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_current_dataset(args.input_csv, args.output_dir)
    print(f"[OK] current_samples_for_lingam.csv -> {paths['csv']}")
    print(f"[OK] current_samples_for_lingam_metadata.json -> {paths['metadata']}")


if __name__ == "__main__":
    main()
