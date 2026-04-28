from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    AXIS_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    GROUP_COLORS,
    SIGN_COLORS,
    TEXT_SECONDARY,
)
from lingam_pipeline_v1.post_attribution.nature_figure_style import apply_nature_style


def load_current_total_effects(total_effect_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(total_effect_csv)
    if "total_effect_abs" not in df.columns:
        raise KeyError("Expected column missing from total effect table: total_effect_abs")
    return df.sort_values(by=["total_effect_abs", "node"], ascending=[False, True]).reset_index(drop=True)


def plot_current_total_effects(
    ax: Axes,
    totals_df: pd.DataFrame,
    x_label: str = "Absolute total effect on risk",
) -> Axes:
    apply_nature_style()
    plot_df = totals_df.sort_values(by=["total_effect_abs", "node"], ascending=[False, True]).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("Current total-effect table is empty.")

    n_items = len(plot_df)
    theta = np.linspace(0.0, 2.0 * np.pi, num=n_items, endpoint=False)
    width = 2.0 * np.pi / n_items * 0.76
    max_effect = float(plot_df["total_effect_abs"].max())
    max_effect = max(max_effect, 1e-9)

    ring_base = 0.28
    ring_height = 0.50
    inner_band_bottom = 0.13
    inner_band_height = 0.08
    radial_scale = plot_df["total_effect_abs"].to_numpy(dtype=float) / max_effect

    ax.set_facecolor(FIGURE_FACE)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    theta_ref = np.linspace(0.0, 2.0 * np.pi, 480)
    for frac in (0.25, 0.50, 0.75, 1.00):
        radius = ring_base + ring_height * frac
        ax.plot(theta_ref, np.full_like(theta_ref, radius), color=GRID_COLOR, linewidth=0.55, zorder=0)
        ax.text(
            np.deg2rad(84.0),
            radius,
            f"{max_effect * frac:.2f}",
            ha="left",
            va="center",
            color=TEXT_SECONDARY,
        )

    ax.bar(
        theta,
        np.full(n_items, inner_band_height),
        width=width * 0.94,
        bottom=inner_band_bottom,
        color=[GROUP_COLORS[group] for group in plot_df["group_lv1"]],
        edgecolor=FIGURE_FACE,
        linewidth=0.5,
        zorder=2,
    )

    ax.bar(
        theta,
        0.09 + radial_scale * ring_height,
        width=width,
        bottom=ring_base,
        color=[GROUP_COLORS[group] for group in plot_df["group_lv1"]],
        edgecolor=FIGURE_FACE,
        linewidth=0.7,
        alpha=0.92,
        zorder=3,
    )

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        tip_r = ring_base + 0.09 + radial_scale[idx] * ring_height
        label_r = tip_r + 0.12
        angle_deg = float(np.degrees(theta[idx]))
        rotation = angle_deg - 90.0
        align = "left"
        if 90.0 < angle_deg < 270.0:
            rotation += 180.0
            align = "right"

        ax.scatter(
            [theta[idx]],
            [tip_r],
            s=30,
            color=SIGN_COLORS.get(row.effect_sign, TEXT_SECONDARY),
            edgecolors=FIGURE_FACE,
            linewidths=0.7,
            zorder=4,
        )
        ax.text(
            theta[idx],
            label_r,
            row.node,
            ha=align,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontweight="bold",
        )

    ax.text(
        0.0,
        0.0,
        "Current\nattribution\nspectrum",
        ha="center",
        va="center",
        color=TEXT_SECONDARY,
    )
    ax.text(
        0.5,
        -0.08,
        x_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT_SECONDARY,
    )

    legend_handles = [
        Patch(facecolor=GROUP_COLORS["Hydrogeology"], edgecolor=AXIS_COLOR, label="Hydrogeology"),
        Patch(facecolor=GROUP_COLORS["Climate"], edgecolor=AXIS_COLOR, label="Climate"),
        Patch(facecolor=GROUP_COLORS["Anthropogenic"], edgecolor=AXIS_COLOR, label="Anthropogenic"),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=4.8,
            markerfacecolor=SIGN_COLORS["positive"],
            markeredgecolor=FIGURE_FACE,
            linestyle="None",
            label="Positive sign",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=4.8,
            markerfacecolor=SIGN_COLORS["negative"],
            markeredgecolor=FIGURE_FACE,
            linestyle="None",
            label="Negative sign",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=2,
        handlelength=1.3,
        columnspacing=0.9,
        borderaxespad=0.2,
    )
    return ax
