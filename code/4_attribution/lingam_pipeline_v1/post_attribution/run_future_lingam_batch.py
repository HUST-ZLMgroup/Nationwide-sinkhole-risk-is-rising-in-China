from __future__ import annotations

import argparse
from pathlib import Path

from lingam_pipeline_v1.post_attribution.run_future_lingam_repeats import write_future_lingam_repeats
from lingam_pipeline_v1.pre_attribution.attribution_config import (
    N_REPEATS,
    OUTPUT_ROOT,
    SAMPLE_RATIO,
    SSPS,
    YEARS,
)
from lingam_pipeline_v1.pre_attribution.build_future_samples_for_lingam import (
    DEFAULT_CLASSIFICATION_ARTIFACTS,
    DEFAULT_PATH_NAME,
    DEFAULT_POINTS_BASE_DIR,
    DEFAULT_PREDICTION_DIR,
    write_future_dataset_for_scenario,
)


DEFAULT_PRIOR_MATRIX = OUTPUT_ROOT / "common" / "prior_knowledge_matrix.csv"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-build future LiNGAM samples and run repeated future LiNGAM across scenarios.")
    parser.add_argument("--ssp", nargs="+", default=["all"])
    parser.add_argument("--year", nargs="+", default=["all"])
    parser.add_argument("--path-name", type=str, default=DEFAULT_PATH_NAME)
    parser.add_argument("--points-base-dir", type=Path, default=DEFAULT_POINTS_BASE_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--classification-artifacts-path", type=Path, default=DEFAULT_CLASSIFICATION_ARTIFACTS)
    parser.add_argument("--prior-matrix-csv", type=Path, default=DEFAULT_PRIOR_MATRIX)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--sample-ratio", type=float, default=SAMPLE_RATIO)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--skip-build-samples",
        action="store_true",
        help="If set, assume future_samples_for_lingam.csv already exists for each scenario.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_ssps = _parse_requested(args.ssp, SSPS, "ssp")
    requested_years = _parse_requested(args.year, YEARS, "year")

    for ssp in requested_ssps:
        for year in requested_years:
            if not args.skip_build_samples:
                sample_paths = write_future_dataset_for_scenario(
                    ssp=ssp,
                    year=year,
                    path_name=args.path_name,
                    points_base_dir=args.points_base_dir,
                    prediction_dir=args.prediction_dir,
                    classification_artifacts_path=args.classification_artifacts_path,
                    output_root=args.output_root / "future",
                )
                print(f"[OK] {ssp}_{year} future_samples_for_lingam.csv -> {sample_paths['csv']}")

            future_csv = args.output_root / "future" / ssp / year / "future_samples_for_lingam.csv"
            repeat_paths = write_future_lingam_repeats(
                ssp=ssp,
                year=year,
                future_csv=future_csv,
                prior_matrix_csv=args.prior_matrix_csv,
                output_dir=args.output_root / "future" / ssp / year,
                n_repeats=args.n_repeats,
                sample_ratio=args.sample_ratio,
                random_state=args.random_state,
            )
            print(f"[OK] {ssp}_{year} edge_records_long.csv -> {repeat_paths['edges']}")
            print(f"[OK] {ssp}_{year} total_effect_records_long.csv -> {repeat_paths['totals']}")


if __name__ == "__main__":
    main()
