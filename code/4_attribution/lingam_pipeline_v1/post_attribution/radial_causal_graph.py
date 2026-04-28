from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    AXIS_COLOR,
    FIGURE_FACE,
    GROUP_COLORS,
    SIGN_COLORS,
    TEXT_SECONDARY,
)
from lingam_pipeline_v1.post_attribution.nature_figure_style import apply_nature_style


CENTER_X = 0.52
CENTER_Y = 0.50
OUTER_RING_R = 0.41
INNER_RING_R = 0.23

NODE_LAYOUT = {
    "DK": (205.0, OUTER_RING_R),
    "DB": (175.0, OUTER_RING_R),
    "DF": (145.0, OUTER_RING_R),
    "HDS": (115.0, OUTER_RING_R),
    "PR": (65.0, OUTER_RING_R),
    "TAS": (92.0, OUTER_RING_R),
    "HUSS": (32.0, OUTER_RING_R),
    "PT": (332.0, OUTER_RING_R),
    "UF": (0.0, OUTER_RING_R),
    "IP": (300.0, OUTER_RING_R),
    "WTD": (240.0, INNER_RING_R),
    "LAI": (295.0, INNER_RING_R),
    "RISK": (0.0, 0.0),
}

NODE_GROUP = {
    "DK": "Hydrogeology",
    "DB": "Hydrogeology",
    "DF": "Hydrogeology",
    "HDS": "Hydrogeology",
    "PR": "Climate",
    "TAS": "Climate",
    "HUSS": "Climate",
    "PT": "Anthropogenic",
    "UF": "Anthropogenic",
    "IP": "Anthropogenic",
    "WTD": "Anthropogenic",
    "LAI": "Anthropogenic",
    "RISK": "Target",
}

SECTOR_SPECS = (
    ("Hydrogeology", 108.0, 220.0, 0.31, 0.49),
    ("Climate", 18.0, 112.0, 0.31, 0.49),
    ("Anthropogenic", 288.0, 360.0, 0.31, 0.49),
    ("Anthropogenic", 0.0, 18.0, 0.31, 0.49),
    ("Mediators", 220.0, 320.0, 0.14, 0.29),
)

SECTOR_LABELS = {
    "Hydrogeology": (166.0, 0.54),
    "Climate": (63.0, 0.54),
    "Human pressure": (335.0, 0.54),
    "Mediators": (270.0, 0.34),
}


def _polar_to_xy(theta_deg: float, radius: float) -> tuple[float, float]:
    theta = math.radians(theta_deg)
    return (
        CENTER_X + radius * math.cos(theta),
        CENTER_Y + radius * math.sin(theta),
    )


def _node_xy(node: str) -> tuple[float, float]:
    theta_deg, radius = NODE_LAYOUT[node]
    if node == "RISK":
        return CENTER_X, CENTER_Y
    return _polar_to_xy(theta_deg, radius)


def _edge_curvature(source: str, target: str) -> float:
    source_angle, _ = NODE_LAYOUT[source]
    if target == "RISK":
        return 0.10 if 20.0 <= source_angle <= 180.0 else -0.10

    target_angle, _ = NODE_LAYOUT[target]
    delta = ((target_angle - source_angle + 180.0) % 360.0) - 180.0
    return float(np.clip(delta / 180.0 * 0.22, -0.22, 0.22))


def _draw_sector_background(ax: Axes) -> None:
    for label, theta1, theta2, r_inner, r_outer in SECTOR_SPECS:
        color_key = "Anthropogenic" if label == "Mediators" else label
        ax.add_patch(
            Wedge(
                center=(CENTER_X, CENTER_Y),
                r=r_outer,
                theta1=theta1,
                theta2=theta2,
                width=r_outer - r_inner,
                facecolor=GROUP_COLORS[color_key],
                edgecolor="none",
                alpha=0.10,
                zorder=0,
            )
        )

    for radius in (OUTER_RING_R, INNER_RING_R):
        ax.add_patch(
            Circle(
                (CENTER_X, CENTER_Y),
                radius=radius,
                facecolor="none",
                edgecolor=AXIS_COLOR,
                linewidth=0.45,
                alpha=0.35,
                linestyle=(0, (1.8, 2.6)),
                zorder=0,
            )
        )

    ax.add_patch(
        Circle(
            (CENTER_X, CENTER_Y),
            radius=0.08,
            facecolor=FIGURE_FACE,
            edgecolor=AXIS_COLOR,
            linewidth=0.7,
            zorder=1,
        )
    )

    for label, (theta_deg, radius) in SECTOR_LABELS.items():
        x_val, y_val = _polar_to_xy(theta_deg, radius)
        ax.text(x_val, y_val, label, ha="center", va="center", color=TEXT_SECONDARY)


def _draw_nodes(ax: Axes, nodes: list[str], center_label: str) -> None:
    for node in nodes:
        x_val, y_val = _node_xy(node)
        if node == "RISK":
            ax.add_patch(
                Circle(
                    (x_val, y_val),
                    radius=0.052,
                    facecolor=GROUP_COLORS["Target"],
                    edgecolor=AXIS_COLOR,
                    linewidth=0.7,
                    zorder=4,
                )
            )
            ax.text(x_val, y_val, center_label, ha="center", va="center", color="white", fontweight="bold", zorder=5)
            continue

        group = NODE_GROUP[node]
        ax.add_patch(
            Circle(
                (x_val, y_val),
                radius=0.028,
                facecolor=GROUP_COLORS[group],
                edgecolor=FIGURE_FACE,
                linewidth=0.9,
                zorder=4,
            )
        )
        ax.text(x_val, y_val, node, ha="center", va="center", zorder=5)


def plot_radial_causal_graph(
    ax: Axes,
    edges_df: pd.DataFrame,
    width_col: str,
    sign_col: str,
    footer_note: str = "",
    center_label: str = "RISK",
    width_scale: tuple[float, float] = (0.8, 3.6),
    alpha_scale: tuple[float, float] = (0.24, 0.78),
    alpha_col: str | None = None,
    show_legend: bool = True,
) -> Axes:
    apply_nature_style()
    required_cols = {"source", "target", width_col, sign_col}
    missing = required_cols.difference(edges_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for radial causal graph: {sorted(missing)}")

    plot_df = edges_df.copy()
    if plot_df.empty:
        raise ValueError("No edges available for radial causal graph.")

    ax.set_facecolor(FIGURE_FACE)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.02, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_sector_background(ax)

    metric = plot_df[width_col].astype(float).to_numpy()
    metric_max = float(np.nanmax(metric)) if len(metric) else 1.0
    metric_max = max(metric_max, 1e-9)

    if alpha_col is None:
        alpha_metric = metric
    else:
        alpha_metric = plot_df[alpha_col].astype(float).to_numpy()
    alpha_max = float(np.nanmax(alpha_metric)) if len(alpha_metric) else 1.0
    alpha_max = max(alpha_max, 1e-9)

    plot_df = plot_df.sort_values(by=[width_col, "source", "target"], ascending=[True, True, True]).reset_index(drop=True)

    width_min, width_max = width_scale
    alpha_min, alpha_max_out = alpha_scale
    for row in plot_df.itertuples(index=False):
        x0, y0 = _node_xy(row.source)
        x1, y1 = _node_xy(row.target)
        line_width = width_min + (width_max - width_min) * float(getattr(row, width_col)) / metric_max
        alpha_value = alpha_min + (alpha_max_out - alpha_min) * float(getattr(row, alpha_col or width_col)) / alpha_max
        arrow = FancyArrowPatch(
            posA=(x0, y0),
            posB=(x1, y1),
            arrowstyle="-|>",
            mutation_scale=7.5 + line_width * 1.1,
            linewidth=line_width,
            color=SIGN_COLORS[getattr(row, sign_col)],
            alpha=alpha_value,
            connectionstyle=f"arc3,rad={_edge_curvature(row.source, row.target)}",
            shrinkA=11.0,
            shrinkB=10.0 if row.target != "RISK" else 15.0,
            zorder=2,
        )
        ax.add_patch(arrow)

    nodes = sorted(set(plot_df["source"]).union(plot_df["target"]), key=lambda node: (NODE_LAYOUT[node][1], NODE_LAYOUT[node][0]))
    if "RISK" not in nodes:
        nodes.append("RISK")
    _draw_nodes(ax, nodes, center_label=center_label)

    if show_legend:
        legend_handles = [
            Line2D([0], [0], color=SIGN_COLORS["positive"], lw=2.0, label="Positive link"),
            Line2D([0], [0], color=SIGN_COLORS["negative"], lw=2.0, label="Negative link"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            frameon=False,
            ncol=2,
            handlelength=1.8,
            columnspacing=1.2,
        )
    if footer_note:
        ax.text(0.04, 0.05, footer_note, transform=ax.transAxes, ha="left", va="center", color=TEXT_SECONDARY)
    return ax
