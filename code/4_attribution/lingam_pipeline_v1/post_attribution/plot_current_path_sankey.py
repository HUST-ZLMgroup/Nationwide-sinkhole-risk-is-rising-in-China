from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes

from lingam_pipeline_v1.post_attribution.radial_causal_graph import plot_radial_causal_graph


def load_current_path_plot_inputs(
    path_summary_csv: str | Path,
    edges_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_df = pd.read_csv(path_summary_csv, keep_default_na=False)
    edges_df = pd.read_csv(edges_csv)
    return paths_df, edges_df


def _selected_edges_from_paths(paths_df: pd.DataFrame, max_paths: int = 18) -> pd.DataFrame:
    selected_paths = (
        paths_df.loc[paths_df["is_main_path"]]
        .sort_values(by=["path_effect_abs", "path_length", "path_str"], ascending=[False, True, True])
        .head(max_paths)
        .copy()
    )
    rows = []
    for row in selected_paths.itertuples(index=False):
        nodes = str(row.path_str).split("->")
        for source, target in zip(nodes[:-1], nodes[1:]):
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "path_effect_abs": float(row.path_effect_abs),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["source", "target", "support_abs"])

    return (
        pd.DataFrame(rows)
        .groupby(["source", "target"], as_index=False)["path_effect_abs"]
        .sum()
        .rename(columns={"path_effect_abs": "support_abs"})
        .sort_values(by=["support_abs", "source", "target"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def plot_current_path_decomposition(
    ax: Axes,
    paths_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_paths: int = 18,
    show_legend: bool = True,
    show_note: bool = True,
) -> Axes:
    edge_support_df = _selected_edges_from_paths(paths_df, max_paths=max_paths)
    if edge_support_df.empty:
        raise ValueError("No main paths were available for plotting.")

    plot_edges = edges_df.merge(edge_support_df, on=["source", "target"], how="inner")
    plot_edges = plot_edges.sort_values(
        by=["support_abs", "edge_weight_abs", "source", "target"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    n_paths = min(max_paths, len(paths_df.loc[paths_df["is_main_path"]]))
    footer_note = f"Top {n_paths} dominant paths; curve width scales with cumulative path support" if show_note else ""
    return plot_radial_causal_graph(
        ax=ax,
        edges_df=plot_edges,
        width_col="support_abs",
        sign_col="edge_sign",
        alpha_col="edge_weight_abs",
        center_label="RISK",
        width_scale=(0.9, 3.8),
        alpha_scale=(0.26, 0.78),
        footer_note=footer_note,
        show_legend=show_legend,
    )
