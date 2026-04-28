from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from lingam_pipeline_v1.pre_attribution.attribution_config import AXIS_COLOR, FIGURE_FACE, GRID_COLOR, OUTPUT_ROOT, TEXT_SECONDARY
from lingam_pipeline_v1.post_attribution.nature_figure_style import apply_nature_style, morandi_sequential_cmap


DEFAULT_HEATMAP_CSV = OUTPUT_ROOT / "future" / "summary" / "edge_frequency_heatmap_matrix.csv"


def _scenario_sort_key(scenario_id: str) -> tuple[int, int, str]:
    try:
        ssp, year = str(scenario_id).split("_", 1)
        return (int(ssp.replace("ssp", "")), int(year), scenario_id)
    except Exception:
        return (999, 9999, str(scenario_id))


def load_edge_frequency_heatmap_matrix(heatmap_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(heatmap_csv)
    scenario_cols = [c for c in df.columns if c != "edge_label"]
    ordered = sorted(scenario_cols, key=_scenario_sort_key)
    return df.loc[:, ["edge_label"] + ordered]


def _draw_vector_colorbar(ax: Axes, norm: Normalize, freq_threshold: float, n_steps: int = 96) -> None:
    cax = ax.inset_axes([1.02, 0.14, 0.05, 0.72])
    cmap = morandi_sequential_cmap()
    for i in range(n_steps):
        y0 = i / n_steps
        y1 = (i + 1) / n_steps
        value = norm.inverse((i + 0.5) / n_steps)
        cax.add_patch(
            Rectangle(
                (0.0, y0),
                1.0,
                y1 - y0,
                facecolor=cmap(norm(value)),
                edgecolor="none",
            )
        )

    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(0.0, 1.0)
    cax.set_xticks([])
    cax.set_yticks(np.linspace(0.0, 1.0, 6))
    cax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(0.0, 1.0, 6)])
    cax.yaxis.tick_right()
    cax.set_ylabel("Selection frequency over 100 repeats")
    cax.yaxis.set_label_position("right")
    cax.axhline(float(freq_threshold), color=AXIS_COLOR, linewidth=0.7, linestyle="--")
    for spine in cax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.6)
    cax.set_facecolor(FIGURE_FACE)


def _short_year_label(scenario_id: str) -> str:
    try:
        return str(scenario_id).split("_", 1)[1][-2:]
    except Exception:
        return str(scenario_id)


def _ssp_label(scenario_id: str) -> str:
    try:
        return str(scenario_id).split("_", 1)[0].upper()
    except Exception:
        return str(scenario_id)


def _scenario_geometry(scenario_cols: list[str], gap_deg: float = 7.0) -> tuple[np.ndarray, float, list[tuple[float, str]]]:
    gap = math.radians(gap_deg)
    unique_ssps = list(dict.fromkeys(_ssp_label(col) for col in scenario_cols))
    n_groups = max(len(unique_ssps), 1)
    cell_width = (2.0 * math.pi - n_groups * gap) / max(len(scenario_cols), 1)

    centers: list[float] = []
    group_centers: list[tuple[float, str]] = []
    angle_cursor = 0.0
    for ssp in unique_ssps:
        group_cols = [col for col in scenario_cols if _ssp_label(col) == ssp]
        group_start = angle_cursor
        for _ in group_cols:
            centers.append(angle_cursor + cell_width / 2.0)
            angle_cursor += cell_width
        group_end = angle_cursor
        group_centers.append(((group_start + group_end) / 2.0, ssp))
        angle_cursor += gap
    return np.asarray(centers, dtype=float), cell_width, group_centers


def _display_edge_subset(heatmap_df: pd.DataFrame, scenario_cols: list[str], top_k_edges: int) -> pd.DataFrame:
    display_df = heatmap_df.copy()
    display_df["mean_freq"] = display_df[scenario_cols].mean(axis=1)
    display_df["std_freq"] = display_df[scenario_cols].std(axis=1)

    core_k = max(4, top_k_edges // 2)
    variable_k = max(4, top_k_edges - core_k)

    core_df = display_df.sort_values(by=["mean_freq", "std_freq", "edge_label"], ascending=[False, False, True]).head(core_k)
    variable_df = (
        display_df.loc[display_df["std_freq"] > 0.02]
        .sort_values(by=["std_freq", "mean_freq", "edge_label"], ascending=[False, False, True])
        .head(variable_k * 2)
    )

    selected_df = pd.concat([variable_df, core_df], ignore_index=True).drop_duplicates(subset=["edge_label"], keep="first")
    if len(selected_df) < top_k_edges:
        filler_df = display_df.sort_values(by=["mean_freq", "std_freq", "edge_label"], ascending=[False, False, True])
        selected_df = pd.concat([selected_df, filler_df], ignore_index=True).drop_duplicates(subset=["edge_label"], keep="first")

    selected_df = selected_df.head(top_k_edges).copy()
    selected_df["cluster_order"] = np.where(selected_df["std_freq"] > 0.02, 0, 1)
    return selected_df.sort_values(by=["cluster_order", "std_freq", "mean_freq"], ascending=[True, False, False]).reset_index(drop=True)


def plot_edge_frequency_heatmap(
    ax: Axes,
    heatmap_df: pd.DataFrame,
    top_k_edges: int = 16,
    freq_threshold: float = 0.8,
) -> Axes:
    apply_nature_style()
    scenario_cols = [c for c in heatmap_df.columns if c != "edge_label"]
    if not scenario_cols:
        raise ValueError("Heatmap input has no scenario columns.")

    plot_df = _display_edge_subset(heatmap_df, scenario_cols=scenario_cols, top_k_edges=top_k_edges)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = morandi_sequential_cmap()

    ax.set_facecolor(FIGURE_FACE)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    centers, cell_width, group_centers = _scenario_geometry(scenario_cols)
    ring_base = 0.28
    ring_height = 0.042
    ring_gap = 0.008
    max_r = ring_base + len(plot_df) * (ring_height + ring_gap) + 0.18
    ax.set_ylim(0.0, max_r)

    for ring_idx, row in enumerate(plot_df.itertuples(index=False)):
        bottom = ring_base + ring_idx * (ring_height + ring_gap)
        values = np.asarray([getattr(row, col) for col in scenario_cols], dtype=float)
        ax.bar(
            centers,
            np.full_like(centers, ring_height),
            width=cell_width * 0.96,
            bottom=bottom,
            color=[cmap(norm(value)) for value in values],
            edgecolor=GRID_COLOR,
            linewidth=0.45,
            align="center",
            zorder=2,
        )
        ax.text(
            math.radians(268.0),
            bottom + ring_height * 0.54,
            str(row.edge_label),
            ha="right",
            va="center",
            color=TEXT_SECONDARY,
        )

    for boundary_idx in range(4, len(scenario_cols), 4):
        boundary_theta = centers[boundary_idx - 1] + cell_width / 2.0
        ax.plot([boundary_theta, boundary_theta], [ring_base - 0.02, max_r - 0.08], color=AXIS_COLOR, linewidth=0.75, zorder=3)

    outer_year_r = ring_base + len(plot_df) * (ring_height + ring_gap) + 0.02
    for theta, scenario_id in zip(centers, scenario_cols):
        angle_deg = np.degrees(theta)
        rotation = angle_deg - 90.0
        align = "left"
        if 90.0 < angle_deg < 270.0:
            rotation += 180.0
            align = "right"
        ax.text(
            theta,
            outer_year_r,
            _short_year_label(scenario_id),
            ha=align,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
        )

    outer_ssp_r = outer_year_r + 0.12
    for theta, ssp_label in group_centers:
        ax.text(theta, outer_ssp_r, ssp_label, ha="center", va="center", fontweight="bold")

    ax.text(0.0, 0.06, "Scenario\nfingerprint\nof recurrent edges", ha="center", va="center", color=TEXT_SECONDARY)
    ax.text(
        math.radians(180.0),
        ring_base - 0.09,
        "Inner rings emphasize scenario-sensitive edges; outer rings retain stable core links",
        ha="center",
        va="center",
        color=TEXT_SECONDARY,
    )

    _draw_vector_colorbar(ax=ax, norm=norm, freq_threshold=freq_threshold)
    return ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the stable-edge frequency heatmap.")
    parser.add_argument("--heatmap-csv", type=Path, default=DEFAULT_HEATMAP_CSV)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k-edges", type=int, default=16)
    parser.add_argument("--freq-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heatmap_df = load_edge_frequency_heatmap_matrix(args.heatmap_csv)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), subplot_kw={"projection": "polar"}, facecolor=FIGURE_FACE)
    plot_edge_frequency_heatmap(ax, heatmap_df, top_k_edges=args.top_k_edges, freq_threshold=args.freq_threshold)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, format="svg", bbox_inches="tight")
        print(f"[OK] edge_frequency_heatmap.svg -> {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
