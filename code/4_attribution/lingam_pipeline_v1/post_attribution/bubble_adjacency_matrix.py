from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    AXIS_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    GROUP_COLORS,
    SIGN_COLORS,
    TEXT_SECONDARY,
)
from lingam_pipeline_v1.post_attribution.nature_figure_style import apply_nature_style


BASE_NODE_ORDER = [
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
    "WTD",
    "LAI",
    "RISK",
]


def node_group(node: str) -> str:
    if node in {"DK", "DB", "DF", "HDS"}:
        return "Hydrogeology"
    if node in {"PR", "TAS", "HUSS"}:
        return "Climate"
    if node == "RISK":
        return "Target"
    return "Anthropogenic"


def semantic_node_order(nodes: list[str] | tuple[str, ...] | np.ndarray) -> list[str]:
    order_map = {name: idx for idx, name in enumerate(BASE_NODE_ORDER)}
    return sorted({str(node) for node in nodes}, key=lambda item: (order_map.get(item, 999), item))


def morandi_diverging_cmap(name: str = "morandi_diverging") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        name,
        [
            SIGN_COLORS["negative"],
            "#E6EAED",
            "#FFFFFF",
            "#F3E7E0",
            SIGN_COLORS["positive"],
        ],
    )


def _value_norm(values: np.ndarray) -> Normalize:
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    max_abs = max(abs(value_min), abs(value_max), 1e-9)
    if value_min < 0.0 < value_max:
        return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    return Normalize(vmin=value_min, vmax=value_max if value_max > value_min else value_min + 1e-9)


def _size_values(values: np.ndarray, size_min: float = 70.0, size_max: float = 940.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_val = max(float(np.nanmax(values)), 1e-9)
    scaled = np.power(np.clip(values / max_val, 0.0, None), 0.9)
    return size_min + (size_max - size_min) * scaled


def _resolve_marker_size_range(
    ax: Axes,
    n_cols: int,
    n_rows: int,
    requested_min: float,
    requested_max: float,
    max_fill: float = 0.82,
) -> tuple[float, float]:
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    if bbox.width <= 0 or bbox.height <= 0 or n_cols <= 0 or n_rows <= 0:
        return requested_min, requested_max

    px_to_pt = 72.0 / ax.figure.dpi
    cell_width_pt = (bbox.width / n_cols) * px_to_pt
    cell_height_pt = (bbox.height / n_rows) * px_to_pt
    max_diameter_pt = max_fill * min(cell_width_pt, cell_height_pt)
    capped_max = np.pi * (max_diameter_pt / 2.0) ** 2
    size_max = min(requested_max, capped_max)
    size_min = min(requested_min, size_max * 0.40)
    size_min = max(size_min, min(26.0, size_max * 0.55))
    return float(size_min), float(size_max)


def _format_support_label(value: float) -> str:
    if value >= 10 and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if value >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _colorbar_ticks(norm: Normalize) -> list[float]:
    if isinstance(norm, TwoSlopeNorm):
        return [float(norm.vmin), 0.0, float(norm.vmax)]
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)
    return [vmin, (vmin + vmax) / 2.0, vmax]


def _support_reference_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return np.asarray([], dtype=float)
    refs = np.asarray(
        [
            float(np.min(values)),
            float(np.quantile(values, 0.55)),
            float(np.max(values)),
        ],
        dtype=float,
    )
    refs = np.unique(np.round(refs, 4))
    return refs


def _add_top_legends(
    ax: Axes,
    norm: Normalize,
    cmap: LinearSegmentedColormap,
    support_values: np.ndarray,
    support_label: str,
    size_min: float,
    size_max: float,
    show_colorbar: bool,
    show_size_legend: bool,
) -> None:
    if show_colorbar:
        color_ax = ax.inset_axes([0.02, 1.22, 0.42, 0.11], transform=ax.transAxes)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        colorbar = plt.colorbar(sm, cax=color_ax, orientation="horizontal")
        colorbar.outline.set_edgecolor(AXIS_COLOR)
        colorbar.outline.set_linewidth(0.55)
        ticks = _colorbar_ticks(norm)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([_format_support_label(tick) for tick in ticks])
        color_ax.tick_params(axis="x", length=2, pad=1)

    refs = _support_reference_values(support_values)
    if not show_size_legend or refs.size == 0:
        return

    size_ax = ax.inset_axes([0.53, 1.15, 0.45, 0.18], transform=ax.transAxes)
    size_ax.set_axis_off()

    legend_sizes = _size_values(refs, size_min=size_min, size_max=size_max)
    if len(refs) == 1:
        x_positions = np.asarray([0.55], dtype=float)
    else:
        x_positions = np.linspace(0.24, 0.86, len(refs))
    y_center = 0.72
    size_ax.scatter(
        x_positions,
        np.full(len(refs), y_center),
        s=legend_sizes,
        facecolor="#DAD6D2",
        edgecolor=AXIS_COLOR,
        linewidth=0.5,
        zorder=3,
    )
    for xpos, value in zip(x_positions, refs):
        size_ax.text(
            xpos,
            0.01,
            _format_support_label(float(value)),
            ha="center",
            va="bottom",
            color="#000000",
        )
    size_ax.set_xlim(0.0, 1.0)
    size_ax.set_ylim(0.0, 1.0)


def plot_bubble_adjacency_matrix(
    ax: Axes,
    edge_df: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    value_col: str = "effect_value",
    support_col: str = "support_value",
    support_label: str = "Support",
    note: str | None = None,
    x_nodes: list[str] | None = None,
    y_nodes: list[str] | None = None,
    show_colorbar: bool = True,
    show_size_legend: bool = True,
) -> Axes:
    apply_nature_style()
    if edge_df.empty:
        raise ValueError("Bubble adjacency matrix received an empty edge table.")

    df = edge_df.copy()
    df[source_col] = df[source_col].astype(str)
    df[target_col] = df[target_col].astype(str)
    df[value_col] = df[value_col].astype(float)
    df[support_col] = df[support_col].astype(float)

    x_nodes = semantic_node_order(df[source_col].unique()) if x_nodes is None else list(x_nodes)
    y_nodes = semantic_node_order(df[target_col].unique()) if y_nodes is None else list(y_nodes)
    x_map = {node: idx for idx, node in enumerate(x_nodes)}
    y_map = {node: idx for idx, node in enumerate(y_nodes)}

    plot_df = df.loc[df[source_col].isin(x_map) & df[target_col].isin(y_map)].copy()
    if plot_df.empty:
        raise ValueError("Bubble adjacency matrix selection removed all edges.")

    norm = _value_norm(plot_df[value_col].to_numpy(dtype=float))
    cmap = morandi_diverging_cmap()

    ax.set_facecolor(FIGURE_FACE)
    ax.set_xlim(-0.5, len(x_nodes) - 0.5)
    ax.set_ylim(len(y_nodes) - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")

    size_min = 72.0
    size_max = 900.0
    size_min, size_max = _resolve_marker_size_range(
        ax,
        n_cols=len(x_nodes),
        n_rows=len(y_nodes),
        requested_min=size_min,
        requested_max=size_max,
        max_fill=0.84,
    )
    sizes = _size_values(plot_df[support_col].to_numpy(dtype=float), size_min=size_min, size_max=size_max)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    span_start = 0
    while span_start < len(x_nodes):
        group_name = node_group(x_nodes[span_start])
        span_end = span_start
        while span_end + 1 < len(x_nodes) and node_group(x_nodes[span_end + 1]) == group_name:
            span_end += 1
        ax.axvspan(
            span_start - 0.5,
            span_end + 0.5,
            facecolor=GROUP_COLORS[group_name],
            alpha=0.07,
            zorder=0,
        )
        ax.text(
            (span_start + span_end) / 2.0,
            -0.30,
            group_name,
            transform=trans,
            ha="center",
            va="top",
            color=TEXT_SECONDARY,
            clip_on=False,
        )
        span_start = span_end + 1

    for x_idx in range(len(x_nodes) + 1):
        ax.axvline(x_idx - 0.5, color=GRID_COLOR, linewidth=0.55, zorder=1)
    for y_idx in range(len(y_nodes) + 1):
        ax.axhline(y_idx - 0.5, color=GRID_COLOR, linewidth=0.55, zorder=1)

    scatter = ax.scatter(
        plot_df[source_col].map(x_map),
        plot_df[target_col].map(y_map),
        s=sizes,
        c=plot_df[value_col],
        cmap=cmap,
        norm=norm,
        edgecolors=AXIS_COLOR,
        linewidths=0.55,
        zorder=3,
    )

    ax.set_xticks(range(len(x_nodes)))
    ax.set_xticklabels(x_nodes, rotation=35, ha="right", rotation_mode="anchor")
    ax.xaxis.tick_bottom()
    ax.tick_params(axis="x", length=0, pad=3)

    ax.set_yticks(range(len(y_nodes)))
    ax.set_yticklabels(y_nodes)
    ax.tick_params(axis="y", length=0, pad=3)

    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.7)

    if note:
        ax.text(0.00, -0.23, note, transform=ax.transAxes, ha="left", va="top", color=TEXT_SECONDARY)

    if show_colorbar:
        _add_top_legends(
            ax=ax,
            norm=norm,
            cmap=cmap,
            support_values=plot_df[support_col].to_numpy(dtype=float),
            support_label=support_label,
            size_min=size_min,
            size_max=size_max,
            show_colorbar=show_colorbar,
            show_size_legend=show_size_legend,
        )
    elif show_size_legend:
        _add_top_legends(
            ax=ax,
            norm=norm,
            cmap=cmap,
            support_values=plot_df[support_col].to_numpy(dtype=float),
            support_label=support_label,
            size_min=size_min,
            size_max=size_max,
            show_colorbar=show_colorbar,
            show_size_legend=show_size_legend,
        )

    return ax


def aggregate_future_bubble_edges(
    edge_frequency_csv: str | Path,
    *,
    year: int | None = None,
    min_support_count: int = 1,
) -> pd.DataFrame:
    df = pd.read_csv(edge_frequency_csv)
    df = df.loc[df["is_stable_edge"]].copy()
    if year is not None:
        df = df.loc[df["year"].astype(int) == int(year)].copy()
    if df.empty:
        raise ValueError("No stable edges remained for future bubble aggregation.")

    max_support = int(df["scenario_id"].nunique())
    agg_df = (
        df.groupby(["source", "target", "source_group", "target_group"], as_index=False)
        .agg(
            support_count=("scenario_id", "nunique"),
            mean_weight=("mean_weight", "mean"),
            mean_weight_abs=("mean_weight_abs", "mean"),
            mean_freq=("freq_present", "mean"),
            mean_freq_positive=("freq_positive", "mean"),
            mean_freq_negative=("freq_negative", "mean"),
        )
    )
    agg_df = agg_df.loc[agg_df["support_count"] >= int(min_support_count)].copy()
    if agg_df.empty:
        raise ValueError("No future edges satisfied the support-count threshold.")

    agg_df["support_fraction"] = agg_df["support_count"] / max(max_support, 1)
    agg_df["effect_value"] = agg_df["mean_weight"]
    agg_df["support_value"] = agg_df["support_fraction"]
    return agg_df.sort_values(
        by=["support_count", "mean_weight_abs", "source", "target"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
