from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from lingam_pipeline_v1.pre_attribution.attribution_config import FIGURE_FACE, OUTPUT_ROOT
from lingam_pipeline_v1.post_attribution.radial_causal_graph import plot_radial_causal_graph


DEFAULT_EDGE_FREQUENCY_CSV = OUTPUT_ROOT / "future" / "summary" / "edge_frequency_summary.csv"


def load_stable_future_edges(edge_frequency_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(edge_frequency_csv)
    return df.loc[df["is_stable_edge"]].copy()


def plot_stable_future_dag(
    ax: Axes,
    stable_edge_df: pd.DataFrame,
    min_stable_scenarios: int = 1,
    top_k_edges: int = 16,
) -> Axes:
    if stable_edge_df.empty:
        raise ValueError("Stable future edge table is empty.")

    agg_full_df = (
        stable_edge_df.groupby(["source", "target", "source_group", "target_group"], as_index=False)
        .agg(
            n_stable_scenarios=("scenario_id", "nunique"),
            mean_weight=("mean_weight", "mean"),
            mean_weight_abs=("mean_weight_abs", "mean"),
            mean_freq=("freq_present", "mean"),
            mean_freq_positive=("freq_positive", "mean"),
            mean_freq_negative=("freq_negative", "mean"),
        )
    )
    max_available_scenarios = int(agg_full_df["n_stable_scenarios"].max()) if len(agg_full_df) else 0

    agg_df = agg_full_df.loc[agg_full_df["n_stable_scenarios"] >= int(min_stable_scenarios)].copy()
    if agg_df.empty:
        raise ValueError("No stable future edges remained after min_stable_scenarios filtering.")

    agg_df["dominant_sign"] = np.where(
        agg_df["mean_freq_positive"] >= agg_df["mean_freq_negative"],
        "positive",
        "negative",
    )
    agg_df = agg_df.sort_values(
        by=["n_stable_scenarios", "mean_weight_abs", "mean_freq", "source", "target"],
        ascending=[False, False, False, True, True],
    ).head(top_k_edges)

    if int(min_stable_scenarios) >= max_available_scenarios and max_available_scenarios > 0:
        note = f"Core links repeated in all {max_available_scenarios} scenario-year combinations"
    else:
        note = f"Links retained in >={int(min_stable_scenarios)} scenario-year combinations"
    note = f"{note}; curve width scales with mean |effect|"

    return plot_radial_causal_graph(
        ax=ax,
        edges_df=agg_df,
        width_col="mean_weight_abs",
        sign_col="dominant_sign",
        alpha_col="n_stable_scenarios",
        center_label="RISK",
        width_scale=(0.9, 3.7),
        alpha_scale=(0.28, 0.78),
        footer_note=note,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the aggregated stable future DAG.")
    parser.add_argument("--edge-frequency-csv", type=Path, default=DEFAULT_EDGE_FREQUENCY_CSV)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-stable-scenarios", type=int, default=20)
    parser.add_argument("--top-k-edges", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stable_edge_df = load_stable_future_edges(args.edge_frequency_csv)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor=FIGURE_FACE)
    plot_stable_future_dag(
        ax,
        stable_edge_df=stable_edge_df,
        min_stable_scenarios=args.min_stable_scenarios,
        top_k_edges=args.top_k_edges,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, format="svg", bbox_inches="tight")
        print(f"[OK] stable_future_dag.svg -> {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
