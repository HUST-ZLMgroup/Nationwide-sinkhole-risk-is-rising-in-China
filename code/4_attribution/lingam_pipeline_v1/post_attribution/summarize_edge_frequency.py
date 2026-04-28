from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    FREQ_THRESHOLD,
    OUTPUT_ROOT,
    SIGN_THRESHOLD,
    SSPS,
    YEARS,
)
from lingam_pipeline_v1.pre_attribution.variable_schema import build_variable_dictionary, node_order


DEFAULT_FUTURE_ROOT = OUTPUT_ROOT / "future"
DEFAULT_PRIOR_MATRIX = OUTPUT_ROOT / "common" / "prior_knowledge_matrix.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "future" / "summary"


def _scenario_sort_key(scenario_id: str) -> tuple[int, int, str]:
    try:
        ssp, year = str(scenario_id).split("_", 1)
        return (SSPS.index(ssp), YEARS.index(year), scenario_id)
    except ValueError:
        return (999, 999, str(scenario_id))
    except IndexError:
        return (999, 999, str(scenario_id))


def _discover_edge_record_paths(future_root: str | Path) -> list[Path]:
    root = Path(future_root)
    return sorted(root.glob("ssp*/[0-9][0-9][0-9][0-9]/edge_records_long.csv"))


def _candidate_edges(prior_matrix_csv: str | Path) -> pd.DataFrame:
    prior_df = pd.read_csv(prior_matrix_csv, index_col=0)
    dictionary = build_variable_dictionary().set_index("node")
    nodes = node_order()
    rows = []
    for target in nodes:
        for source in nodes:
            if source == target:
                continue
            prior_value = int(prior_df.loc[target, source])
            if prior_value == 0:
                continue
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "source_group": dictionary.loc[source, "group_lv1"],
                    "target_group": dictionary.loc[target, "group_lv1"],
                    "edge_label": f"{source} -> {target}",
                }
            )
    return pd.DataFrame(rows)


def build_edge_frequency_summary(
    future_root: str | Path = DEFAULT_FUTURE_ROOT,
    prior_matrix_csv: str | Path = DEFAULT_PRIOR_MATRIX,
    freq_threshold: float = FREQ_THRESHOLD,
    sign_threshold: float = SIGN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_paths = _discover_edge_record_paths(future_root)
    if not edge_paths:
        raise FileNotFoundError(f"No edge_records_long.csv files found under: {future_root}")

    edge_df = pd.concat([pd.read_csv(path) for path in edge_paths], ignore_index=True)
    if edge_df.empty:
        raise ValueError("Discovered edge_records_long.csv files, but all were empty.")

    scenario_info = (
        edge_df[["scenario_id", "ssp", "year", "repeat_id"]]
        .drop_duplicates()
        .groupby(["scenario_id", "ssp", "year"], as_index=False)
        .agg(n_repeats=("repeat_id", "nunique"))
    )

    candidates = _candidate_edges(prior_matrix_csv)
    base = scenario_info.merge(candidates, how="cross")

    edge_stats = (
        edge_df.groupby(
            ["scenario_id", "ssp", "year", "source", "target", "source_group", "target_group"],
            as_index=False,
        )
        .agg(
            n_present=("edge_present", "sum"),
            n_positive=("edge_sign", lambda s: int((s == "positive").sum())),
            n_negative=("edge_sign", lambda s: int((s == "negative").sum())),
            mean_weight=("edge_weight", "mean"),
            mean_weight_abs=("edge_weight_abs", "mean"),
            median_weight=("edge_weight", "median"),
            sd_weight=("edge_weight", "std"),
        )
    )

    summary_df = base.merge(
        edge_stats,
        on=["scenario_id", "ssp", "year", "source", "target", "source_group", "target_group"],
        how="left",
    )
    summary_df["n_present"] = summary_df["n_present"].fillna(0).astype(int)
    summary_df["n_positive"] = summary_df["n_positive"].fillna(0).astype(int)
    summary_df["n_negative"] = summary_df["n_negative"].fillna(0).astype(int)
    for col in ["mean_weight", "mean_weight_abs", "median_weight", "sd_weight"]:
        summary_df[col] = summary_df[col].fillna(0.0)

    summary_df["freq_present"] = summary_df["n_present"] / summary_df["n_repeats"]
    summary_df["freq_positive"] = summary_df["n_positive"] / summary_df["n_repeats"]
    summary_df["freq_negative"] = summary_df["n_negative"] / summary_df["n_repeats"]
    summary_df["sign_majority_share_present"] = 0.0
    present_mask = summary_df["n_present"] > 0
    summary_df.loc[present_mask, "sign_majority_share_present"] = (
        summary_df.loc[present_mask, ["n_positive", "n_negative"]].max(axis=1)
        / summary_df.loc[present_mask, "n_present"]
    )
    summary_df["passes_freq_threshold"] = summary_df["freq_present"] >= float(freq_threshold)
    summary_df["passes_sign_stability"] = summary_df["sign_majority_share_present"] >= float(sign_threshold)
    summary_df["is_stable_edge"] = summary_df["passes_freq_threshold"] & summary_df["passes_sign_stability"]

    summary_df = summary_df.sort_values(
        by=["scenario_id", "freq_present", "mean_weight_abs", "source", "target"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    scenario_order = sorted(summary_df["scenario_id"].drop_duplicates().tolist(), key=_scenario_sort_key)
    heatmap_df = (
        summary_df.pivot_table(
            index="edge_label",
            columns="scenario_id",
            values="freq_present",
            fill_value=0.0,
            aggfunc="first",
        )
        .reindex(columns=scenario_order)
        .reset_index()
    )
    return summary_df, heatmap_df


def write_edge_frequency_summary(
    future_root: str | Path = DEFAULT_FUTURE_ROOT,
    prior_matrix_csv: str | Path = DEFAULT_PRIOR_MATRIX,
    output_dir: str | Path | None = None,
    freq_threshold: float = FREQ_THRESHOLD,
    sign_threshold: float = SIGN_THRESHOLD,
) -> dict[str, Path]:
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, heatmap_df = build_edge_frequency_summary(
        future_root=future_root,
        prior_matrix_csv=prior_matrix_csv,
        freq_threshold=freq_threshold,
        sign_threshold=sign_threshold,
    )
    summary_path = out_dir / "edge_frequency_summary.csv"
    heatmap_path = out_dir / "edge_frequency_heatmap_matrix.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    heatmap_df.to_csv(heatmap_path, index=False, encoding="utf-8-sig")
    return {"summary": summary_path, "heatmap": heatmap_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stable-edge frequencies from repeated future-stage LiNGAM outputs.")
    parser.add_argument(
        "--future-root",
        type=Path,
        default=DEFAULT_FUTURE_ROOT,
        help="Root directory that contains outputs/future/{ssp}/{year}/edge_records_long.csv files.",
    )
    parser.add_argument(
        "--prior-matrix-csv",
        type=Path,
        default=DEFAULT_PRIOR_MATRIX,
        help="Prior knowledge matrix CSV used to define the candidate edge universe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregated edge-frequency outputs.",
    )
    parser.add_argument("--freq-threshold", type=float, default=FREQ_THRESHOLD)
    parser.add_argument("--sign-threshold", type=float, default=SIGN_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_edge_frequency_summary(
        future_root=args.future_root,
        prior_matrix_csv=args.prior_matrix_csv,
        output_dir=args.output_dir,
        freq_threshold=args.freq_threshold,
        sign_threshold=args.sign_threshold,
    )
    print(f"[OK] edge_frequency_summary.csv -> {paths['summary']}")
    print(f"[OK] edge_frequency_heatmap_matrix.csv -> {paths['heatmap']}")


if __name__ == "__main__":
    main()
