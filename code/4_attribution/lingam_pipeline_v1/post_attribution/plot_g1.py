from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    AXIS_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    GROUP_COLORS,
    OUTPUT_ROOT,
    SIGN_COLORS,
    TEXT_SECONDARY,
)
from lingam_pipeline_v1.post_attribution.bubble_adjacency_matrix import (
    node_group,
    plot_bubble_adjacency_matrix,
    semantic_node_order,
)
from lingam_pipeline_v1.post_attribution.nature_figure_style import apply_nature_style
from lingam_pipeline_v1.post_attribution.plot_current_path_sankey import load_current_path_plot_inputs
from lingam_pipeline_v1.post_attribution.plot_current_total_effects import load_current_total_effects


CURRENT_TOTAL_EFFECT_CSV = OUTPUT_ROOT / "current" / "current_total_effects_to_target.csv"
CURRENT_PATH_SUMMARY_CSV = OUTPUT_ROOT / "current" / "current_path_summary.csv"
CURRENT_EDGES_CSV = OUTPUT_ROOT / "current" / "current_lingam_edges_long.csv"
DEFAULT_OUT = OUTPUT_ROOT / "figures" / "final" / "group1.svg"


def _add_panel_label(fig, ax, label: str) -> None:
    bbox = ax.get_position()
    fig.text(bbox.x0 - 0.016, bbox.y1 + 0.008, label, ha="left", va="bottom", fontweight="bold", fontsize=14)


def _load_current_matrix_edges(
    current_edges_csv: str | Path,
    top_k_edges: int = 24,
    required_sources: tuple[str, ...] = ("DF",),
    required_edges_per_source: int = 2,
) -> pd.DataFrame:
    df = pd.read_csv(current_edges_csv)
    df["priority"] = (
        2 * df["is_selected_for_main_figure"].astype(int)
        + df["is_direct_to_target"].astype(int)
        + 0.25 * df["is_mediator_edge"].astype(int)
    )
    ranked_df = df.sort_values(
        by=["priority", "edge_weight_abs", "source", "target"],
        ascending=[False, False, True, True],
    )
    plot_df = ranked_df.head(top_k_edges).copy()

    if required_sources:
        pinned_df = (
            ranked_df.loc[ranked_df["source"].isin(required_sources)]
            .groupby("source", as_index=False, group_keys=False)
            .head(required_edges_per_source)
            .copy()
        )
        plot_df = (
            pd.concat([plot_df, pinned_df], ignore_index=True)
            .drop_duplicates(subset=["source", "target"], keep="first")
            .copy()
        )

    plot_df["effect_value"] = plot_df["edge_weight"]
    plot_df["support_value"] = plot_df["edge_weight_abs"]
    return plot_df


def _plot_group_facet(ax, totals_df: pd.DataFrame, group_name: str, top_k: int = 3) -> None:
    apply_nature_style()
    group_df = totals_df.loc[totals_df["group_lv1"] == group_name].sort_values(
        by=["total_effect_abs", "node"],
        ascending=[False, True],
    )
    n_rows = len(group_df) if top_k is None else int(top_k)
    plot_df = group_df.head(n_rows).iloc[::-1].reset_index(drop=True)
    max_val = max(float(plot_df["total_effect_abs"].max()), 1e-9)
    ax.set_facecolor(FIGURE_FACE)
    ax.axvspan(0.0, max_val * 1.18, color=GROUP_COLORS[group_name], alpha=0.08, zorder=0)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.45)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    for idx, row in plot_df.iterrows():
        ax.hlines(idx, 0.0, row["total_effect_abs"], color=GROUP_COLORS[group_name], linewidth=2.0)
        ax.scatter(row["total_effect_abs"], idx, s=48, color=GROUP_COLORS[group_name], edgecolors=FIGURE_FACE, linewidths=0.8, zorder=3)
        ax.scatter(row["total_effect_abs"], idx, s=18, color=SIGN_COLORS[row["effect_sign"]], edgecolors="none", zorder=4)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["node"])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0.0, max_val * 1.20)
    ax.text(0.02, 0.90, group_name, transform=ax.transAxes, ha="left", va="top", color=GROUP_COLORS[group_name], fontweight="bold")


def _selected_edges_from_paths(paths_df: pd.DataFrame, max_paths: int = 14) -> pd.DataFrame:
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
                    "support_abs": float(row.path_effect_abs),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["source", "target", "support_abs"])
    return (
        pd.DataFrame(rows)
        .groupby(["source", "target"], as_index=False)["support_abs"]
        .sum()
        .sort_values(by=["support_abs", "source", "target"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _scale_values(values: pd.Series, out_min: float, out_max: float) -> pd.Series:
    values = values.astype(float)
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        return pd.Series(np.full(len(values), (out_min + out_max) / 2.0), index=values.index)
    scaled = (values - vmin) / (vmax - vmin)
    return out_min + scaled * (out_max - out_min)


def _circle_layout(nodes: list[str], radius: float = 1.0) -> tuple[dict[str, tuple[float, float]], dict[str, float], float]:
    ordered = semantic_node_order(nodes)
    n_nodes = max(len(ordered), 1)
    angle_step = 2.0 * np.pi / n_nodes
    start_angle = -np.pi / 2.0 + angle_step * (n_nodes - 1)
    positions: dict[str, tuple[float, float]] = {}
    angles: dict[str, float] = {}
    for idx, node in enumerate(ordered):
        angle = start_angle - idx * angle_step
        positions[node] = (radius * float(np.cos(angle)), radius * float(np.sin(angle)))
        angles[node] = angle
    return positions, angles, angle_step


def _support_guide(ax) -> None:
    y_main = -1.14
    y_weight = -1.28
    ax.plot([-1.12, -0.96], [y_main, y_main], color=SIGN_COLORS["positive"], linewidth=1.8, solid_capstyle="round", clip_on=False)
    ax.text(-0.92, y_main, "Promotes disaster", ha="left", va="center", color="#000000")
    ax.plot([0.02, 0.18], [y_main, y_main], color=SIGN_COLORS["negative"], linewidth=1.8, solid_capstyle="round", clip_on=False)
    ax.text(0.22, y_main, "Reduces disaster", ha="left", va="center", color="#000000")
    ax.plot([0.18, 0.34], [y_weight, y_weight], color=AXIS_COLOR, linewidth=1.0, solid_capstyle="round", alpha=0.8, clip_on=False)
    ax.plot([0.43, 0.59], [y_weight, y_weight], color=AXIS_COLOR, linewidth=3.1, solid_capstyle="round", alpha=0.8, clip_on=False)
    ax.text(0.68, y_weight, "Cumulative path weight", ha="left", va="center", color=TEXT_SECONDARY)


def _plot_circular_path_network(
    ax,
    paths_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_paths: int = 14,
) -> None:
    apply_nature_style()
    edge_support_df = _selected_edges_from_paths(paths_df, max_paths=max_paths)
    plot_edges = edges_df.merge(edge_support_df, on=["source", "target"], how="inner")
    plot_edges = plot_edges.sort_values(
        by=["support_abs", "edge_weight_abs", "source", "target"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    if plot_edges.empty:
        raise ValueError("No dominant-path edges were available for circular plotting.")

    all_nodes = semantic_node_order(pd.concat([edges_df["source"], edges_df["target"]], ignore_index=True).dropna().astype(str).unique())
    positions, angles, angle_step = _circle_layout(all_nodes, radius=1.04)
    group_sequence = [node_group(node) for node in all_nodes]

    ax.set_facecolor(FIGURE_FACE)
    ax.set_aspect("equal")
    ax.axis("off")

    outer_ring = Circle((0.0, 0.0), radius=1.04, facecolor="none", edgecolor=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.add_patch(outer_ring)

    start_idx = 0
    while start_idx < len(all_nodes):
        group_name = group_sequence[start_idx]
        end_idx = start_idx
        while end_idx + 1 < len(all_nodes) and group_sequence[end_idx + 1] == group_name:
            end_idx += 1

        theta_start = np.degrees(angles[all_nodes[end_idx]] - angle_step / 2.0)
        theta_end = np.degrees(angles[all_nodes[start_idx]] + angle_step / 2.0)
        if theta_end <= theta_start:
            theta_end += 360.0
        ax.add_patch(
            Wedge(
                center=(0.0, 0.0),
                r=1.14,
                theta1=theta_start,
                theta2=theta_end,
                width=0.08,
                facecolor=GROUP_COLORS[group_name],
                edgecolor="none",
                alpha=0.18 if group_name != "Target" else 0.24,
                zorder=0,
            )
        )
        start_idx = end_idx + 1

    node_support = pd.concat(
        [
            plot_edges.groupby("source", as_index=False)["support_abs"].sum().rename(columns={"source": "node"}),
            plot_edges.groupby("target", as_index=False)["support_abs"].sum().rename(columns={"target": "node"}),
        ],
        ignore_index=True,
    ).groupby("node", as_index=False)["support_abs"].sum()
    node_support_map = dict(zip(node_support["node"], node_support["support_abs"]))
    active_values = pd.Series({node: node_support_map.get(node, 0.0) for node in all_nodes})
    node_sizes = _scale_values(active_values.where(active_values > 0.0, 0.02), 120.0, 340.0)

    width_map = _scale_values(plot_edges["support_abs"], 0.9, 3.4)
    alpha_map = _scale_values(plot_edges["edge_weight_abs"], 0.36, 0.84)

    for edge_idx, row in plot_edges.iterrows():
        x1, y1 = positions[row["source"]]
        x2, y2 = positions[row["target"]]
        angle_diff = (angles[row["target"]] - angles[row["source"]] + np.pi) % (2.0 * np.pi) - np.pi
        curvature = 0.12 + 0.16 * min(abs(angle_diff) / np.pi, 1.0)
        curvature *= -1.0 if angle_diff > 0.0 else 1.0
        if row["target"] == "RISK":
            curvature *= 0.78

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={curvature:.3f}",
            arrowstyle="-|>",
            mutation_scale=7.5 + float(width_map.iloc[edge_idx]) * 1.5,
            shrinkA=9.0,
            shrinkB=10.0,
            linewidth=float(width_map.iloc[edge_idx]),
            color=SIGN_COLORS[row["edge_sign"]],
            alpha=float(alpha_map.iloc[edge_idx]),
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
        ax.add_patch(patch)

    for node in all_nodes:
        x, y = positions[node]
        group_name = node_group(node)
        is_active = active_values.loc[node] > 0.0
        ax.scatter(
            x,
            y,
            s=float(node_sizes.loc[node]),
            color=GROUP_COLORS[group_name],
            alpha=0.96 if is_active else 0.32,
            edgecolors=AXIS_COLOR,
            linewidths=0.7,
            zorder=3,
        )

        label_r = 1.23 if node != "RISK" else 1.18
        lx = label_r * float(np.cos(angles[node]))
        ly = label_r * float(np.sin(angles[node]))
        ha = "center"
        if lx > 0.10:
            ha = "left"
        elif lx < -0.10:
            ha = "right"
        ax.text(
            lx,
            ly,
            node,
            ha=ha,
            va="center",
            color=TEXT_SECONDARY if not is_active else "#4E4A46",
            alpha=0.94 if is_active else 0.55,
        )

    _support_guide(ax)
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.40, 1.22)


def save_group1(out_path: str | Path = DEFAULT_OUT) -> Path:
    apply_nature_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    totals_df = load_current_total_effects(CURRENT_TOTAL_EFFECT_CSV)
    paths_df, current_edges_df = load_current_path_plot_inputs(CURRENT_PATH_SUMMARY_CSV, CURRENT_EDGES_CSV)
    matrix_df = _load_current_matrix_edges(CURRENT_EDGES_CSV, top_k_edges=24)
    x_nodes = semantic_node_order(matrix_df["source"].unique())
    y_nodes = semantic_node_order(matrix_df["target"].unique())

    fig = plt.figure(figsize=(7.18, 5.56), facecolor=FIGURE_FACE)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.74, 1.26],
        height_ratios=[0.70, 1.10],
        left=0.062,
        right=0.982,
        top=0.962,
        bottom=0.090,
        hspace=0.10,
        wspace=0.10,
    )
    facet_groups = ["Hydrogeology", "Climate", "Anthropogenic"]
    facet_topk = {
        group_name: int(totals_df.loc[totals_df["group_lv1"] == group_name].shape[0])
        for group_name in facet_groups
    }
    left_gs = gs[:, 0].subgridspec(
        3,
        1,
        hspace=0.12,
        height_ratios=[facet_topk[group_name] for group_name in facet_groups],
    )
    facet_axes = [fig.add_subplot(left_gs[i, 0]) for i in range(3)]
    for ax, group_name in zip(facet_axes, facet_groups):
        _plot_group_facet(ax, totals_df, group_name, top_k=facet_topk[group_name])
    facet_axes[-1].set_xlabel("Absolute total effect")

    ax_rt = fig.add_subplot(gs[0, 1])
    plot_bubble_adjacency_matrix(
        ax_rt,
        matrix_df,
        support_label="|β|",
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        show_colorbar=True,
        show_size_legend=True,
    )

    ax_rb = fig.add_subplot(gs[1, 1])
    _plot_circular_path_network(
        ax_rb,
        paths_df=paths_df,
        edges_df=current_edges_df,
        max_paths=14,
    )

    _add_panel_label(fig, facet_axes[0], "a")
    _add_panel_label(fig, ax_rt, "b")
    _add_panel_label(fig, ax_rb, "c")

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final group1 figure.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = save_group1(args.out)
    print(f"[OK] group1 -> {out}")


if __name__ == "__main__":
    main()
