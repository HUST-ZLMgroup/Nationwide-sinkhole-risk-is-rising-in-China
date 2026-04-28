from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    AXIS_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    GROUP_COLORS,
    OUTPUT_ROOT,
    SIGN_COLORS,
    TEXT_SECONDARY,
)
from lingam_pipeline_v1.post_attribution.nature_figure_style import apply_nature_style


CURRENT_TOTAL_EFFECT_CSV = OUTPUT_ROOT / "current" / "current_total_effects_to_target.csv"
CURRENT_PATH_SUMMARY_CSV = OUTPUT_ROOT / "current" / "current_path_summary.csv"
CURRENT_EDGES_CSV = OUTPUT_ROOT / "current" / "current_lingam_edges_long.csv"
FUTURE_STABLE_PATH_CSV = OUTPUT_ROOT / "future" / "summary" / "stable_path_summary.csv"
FUTURE_EDGE_FREQUENCY_CSV = OUTPUT_ROOT / "future" / "summary" / "edge_frequency_summary.csv"
DEFAULT_OUT = OUTPUT_ROOT / "figures" / "final" / "group2.svg"
SUMMARY_DIR = OUTPUT_ROOT / "summary" / "final"

ROLE_X = {
    "DK": 0.10,
    "DB": 0.10,
    "DF": 0.10,
    "HDS": 0.10,
    "PR": 0.34,
    "TAS": 0.34,
    "HUSS": 0.34,
    "PT": 0.58,
    "UF": 0.58,
    "IP": 0.58,
    "WTD": 0.80,
    "LAI": 0.80,
    "RISK": 0.94,
}

ROLE_ORDER = {
    0.10: ["DK", "DB", "DF", "HDS"],
    0.34: ["PR", "TAS", "HUSS"],
    0.58: ["PT", "UF", "IP"],
    0.80: ["WTD", "LAI"],
    0.94: ["RISK"],
}

COLUMN_HEADERS = {
    0.10: "Hydrogeology",
    0.34: "Climate",
    0.58: "Human pressure",
    0.80: "Mediators",
    0.94: "Risk",
}

SOURCE_GROUP_ORDER = ["Hydrogeology", "Climate", "Anthropogenic"]
CHANNEL_ORDER = ["Direct", "via LAI", "via UF", "via WTD", "Serial"]
CHANNEL_LABEL_TEXT = {
    "Direct": "Direct",
    "via LAI": "LAI-mediated",
    "via UF": "UF-mediated",
    "via WTD": "WTD-mediated",
    "Serial": "Multi-step",
}
VARIABLE_ORDER = ["HDS", "DK", "DB", "DF", "HUSS", "TAS", "PR", "PT", "LAI", "UF", "WTD", "IP"]


def _add_panel_label(fig, ax, label: str) -> None:
    bbox = ax.get_position()
    fig.text(bbox.x0 - 0.016, bbox.y1 + 0.008, label, ha="left", va="bottom", fontweight="bold", fontsize=14)


def _node_group(node: str) -> str:
    if node == "RISK":
        return "Target"
    if node in {"DK", "DB", "DF", "HDS"}:
        return "Hydrogeology"
    if node in {"PR", "TAS", "HUSS"}:
        return "Climate"
    return "Anthropogenic"


def _layout_nodes_rect(nodes: list[str]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for x_val, ordered_nodes in ROLE_ORDER.items():
        present = [node for node in ordered_nodes if node in nodes]
        if not present:
            continue
        y_values = np.linspace(0.82, 0.18, num=len(present))
        for node, y_val in zip(present, y_values):
            positions[node] = (x_val, float(y_val))
    return positions


def _edge_rad_rect(source_xy: tuple[float, float], target_xy: tuple[float, float]) -> float:
    _, y0 = source_xy
    _, y1 = target_xy
    dy = y1 - y0
    if abs(dy) < 0.04:
        return 0.0
    return float(np.clip(dy * 0.35, -0.22, 0.22))


def _compact_path_label(path_str: str) -> str:
    nodes = ["R" if node == "RISK" else node for node in str(path_str).split("->")]
    return "-".join(nodes)


def _path_channel(path_str: str) -> str:
    nodes = str(path_str).split("->")
    if len(nodes) <= 2:
        return "Direct"
    inner = nodes[1:-1]
    if len(inner) > 1:
        return "Serial"
    if inner[0] == "LAI":
        return "via LAI"
    if inner[0] == "UF":
        return "via UF"
    if inner[0] == "WTD":
        return "via WTD"
    return "Serial"


def _retention_class(scenario_count: int, max_scenarios: int) -> str:
    if scenario_count >= max_scenarios:
        return "Full"
    if scenario_count > 0:
        return "Conditional"
    return "Lost"


def build_variable_impact_table() -> pd.DataFrame:
    current_df = pd.read_csv(CURRENT_TOTAL_EFFECT_CSV)
    current_df = current_df.loc[:, ["node", "group_lv1", "total_effect", "total_effect_abs"]].copy()
    current_df = current_df.rename(
        columns={
            "total_effect": "historical_effect",
            "total_effect_abs": "historical_abs",
        }
    )

    future_files = sorted((OUTPUT_ROOT / "future").glob("ssp*/[0-9][0-9][0-9][0-9]/total_effect_records_long.csv"))
    if not future_files:
        raise FileNotFoundError("No future total_effect_records_long.csv files were found for building Group 2 panel b.")

    future_frames = [
        pd.read_csv(
            path,
            usecols=["scenario_id", "node", "group_lv1", "total_effect", "total_effect_abs"],
        )
        for path in future_files
    ]
    future_long = pd.concat(future_frames, ignore_index=True)
    scenario_means = (
        future_long.groupby(["scenario_id", "node", "group_lv1"], as_index=False)
        .agg(
            future_effect=("total_effect", "mean"),
            future_abs=("total_effect_abs", "mean"),
        )
        .reset_index(drop=True)
    )
    future_df = (
        scenario_means.groupby(["node", "group_lv1"], as_index=False)
        .agg(
            future_effect=("future_effect", "mean"),
            future_abs=("future_abs", "mean"),
        )
        .reset_index(drop=True)
    )

    impact_df = current_df.merge(future_df, on=["node", "group_lv1"], how="left")
    impact_df["future_effect"] = impact_df["future_effect"].fillna(0.0)
    impact_df["future_abs"] = impact_df["future_abs"].fillna(0.0)
    impact_df["delta_abs"] = impact_df["future_abs"] - impact_df["historical_abs"]
    impact_df["delta_sign"] = np.where(
        impact_df["delta_abs"] > 0,
        "positive",
        np.where(impact_df["delta_abs"] < 0, "negative", "zero"),
    )
    order_map = {node: idx for idx, node in enumerate(VARIABLE_ORDER)}
    return impact_df.sort_values(by=["node"], key=lambda s: s.map(order_map)).reset_index(drop=True)


def _weighted_spans(items: list[str], weights: dict[str, float], angle_start: float, angle_end: float, gap_deg: float) -> dict[str, tuple[float, float, float]]:
    total_weight = sum(max(weights.get(item, 0.0), 1e-6) for item in items)
    usable = max(angle_end - angle_start - gap_deg * max(len(items) - 1, 0), 1e-6)
    out: dict[str, tuple[float, float, float]] = {}
    cursor = angle_start
    for item in items:
        span = usable * max(weights.get(item, 0.0), 1e-6) / total_weight
        out[item] = (cursor, cursor + span, cursor + span / 2.0)
        cursor += span + gap_deg
    return out


def _polar_to_xy(theta_deg: float, radius: float, center: tuple[float, float] = (0.5, 0.5)) -> tuple[float, float]:
    theta = np.deg2rad(theta_deg)
    return center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)


def build_retention_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    current_paths = pd.read_csv(CURRENT_PATH_SUMMARY_CSV, keep_default_na=False)
    current_paths = current_paths.loc[current_paths["is_main_path"]].copy()

    future_paths = pd.read_csv(FUTURE_STABLE_PATH_CSV, keep_default_na=False)
    max_scenarios = int(future_paths["scenario_id"].nunique())
    future_agg = (
        future_paths.groupby("path_str", as_index=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            future_mean_abs=("mean_path_effect_abs", "mean"),
            future_mean_signed=("mean_path_effect", "mean"),
            source_group=("source_group", "first"),
        )
        .reset_index(drop=True)
    )

    retention_df = current_paths[
        ["path_str", "path_effect_abs", "source_group"]
    ].merge(future_agg, on=["path_str", "source_group"], how="left")
    retention_df["scenario_count"] = retention_df["scenario_count"].fillna(0).astype(int)
    retention_df["future_mean_abs"] = retention_df["future_mean_abs"].fillna(0.0)
    retention_df["future_mean_signed"] = retention_df["future_mean_signed"].fillna(0.0)
    retention_df["future_channel"] = retention_df["path_str"].map(_path_channel)
    retention_df["retention_class"] = retention_df["scenario_count"].map(lambda n: _retention_class(int(n), max_scenarios))
    retention_df = retention_df.sort_values(
        by=["scenario_count", "path_effect_abs", "path_str"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    current_path_set = set(retention_df["path_str"])
    future_only_df = future_agg.loc[~future_agg["path_str"].isin(current_path_set)].copy()
    future_only_df["future_channel"] = future_only_df["path_str"].map(_path_channel)
    future_only_df = future_only_df.sort_values(
        by=["scenario_count", "future_mean_abs", "path_str"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    current_edges = pd.read_csv(CURRENT_EDGES_CSV)
    current_edges = current_edges.loc[current_edges["is_selected_for_main_figure"]].copy()

    future_edges = pd.read_csv(FUTURE_EDGE_FREQUENCY_CSV)
    future_edges = future_edges.loc[future_edges["is_stable_edge"]].copy()
    future_edge_agg = (
        future_edges.groupby(["source", "target"], as_index=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            future_mean_abs=("mean_weight_abs", "mean"),
            future_mean_signed=("mean_weight", "mean"),
            source_group=("source_group", "first"),
            target_group=("target_group", "first"),
        )
        .reset_index(drop=True)
    )
    current_edge_retention_df = current_edges[
        ["source", "target", "edge_weight_abs", "source_group", "target_group"]
    ].merge(future_edge_agg, on=["source", "target", "source_group", "target_group"], how="left")
    current_edge_retention_df["scenario_count"] = current_edge_retention_df["scenario_count"].fillna(0).astype(int)
    current_edge_retention_df["future_mean_abs"] = current_edge_retention_df["future_mean_abs"].fillna(0.0)
    current_edge_retention_df["future_mean_signed"] = current_edge_retention_df["future_mean_signed"].fillna(0.0)
    current_edge_retention_df["retention_class"] = current_edge_retention_df["scenario_count"].map(lambda n: _retention_class(int(n), max_scenarios))
    current_edge_retention_df["future_sign"] = np.where(current_edge_retention_df["future_mean_signed"] >= 0, "positive", "negative")
    current_edge_retention_df = current_edge_retention_df.sort_values(
        by=["scenario_count", "edge_weight_abs", "source", "target"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    current_edge_pairs = set(zip(current_edge_retention_df["source"], current_edge_retention_df["target"]))
    future_only_edge_df = future_edge_agg.loc[
        ~future_edge_agg.apply(lambda row: (row["source"], row["target"]) in current_edge_pairs, axis=1)
    ].copy()
    future_only_edge_df["future_sign"] = np.where(future_only_edge_df["future_mean_signed"] >= 0, "positive", "negative")
    future_only_edge_df = future_only_edge_df.sort_values(
        by=["scenario_count", "future_mean_abs", "source", "target"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    meta = {
        "current_main_paths": int(len(retention_df)),
        "full_retained": int((retention_df["retention_class"] == "Full").sum()),
        "conditional": int((retention_df["retention_class"] == "Conditional").sum()),
        "lost": int((retention_df["retention_class"] == "Lost").sum()),
        "future_only": int(len(future_only_df)),
        "n_scenarios": max_scenarios,
    }
    return retention_df, future_only_df, current_edge_retention_df, future_only_edge_df, meta


def save_summary(summary_dir: Path, retention_df: pd.DataFrame, future_only_df: pd.DataFrame, meta: dict[str, int]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    retention_df.to_csv(summary_dir / "retained.csv", index=False)
    future_only_df.to_csv(summary_dir / "added.csv", index=False)
    (summary_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# Retention Summary",
        "",
        f"- Current main paths: {meta['current_main_paths']}",
        f"- Full retention: {meta['full_retained']}/{meta['n_scenarios']}",
        f"- Conditional retention: {meta['conditional']}",
        f"- Lost paths: {meta['lost']}",
        f"- Future-only stable paths: {meta['future_only']}",
        "",
        "Future attribution preserves the current core scaffold and mainly expands mediated routes through `UF`, `LAI`, and `WTD`.",
    ]
    (summary_dir / "retention.md").write_text("\n".join(lines), encoding="utf-8")


def _select_paths_for_dumbbell(retention_df: pd.DataFrame, max_items: int = 10) -> pd.DataFrame:
    special_df = retention_df.loc[retention_df["retention_class"] != "Full"].copy()
    remain_df = retention_df.loc[retention_df["retention_class"] == "Full"].copy()
    remain_df = remain_df.sort_values(by=["path_effect_abs", "future_mean_abs", "path_str"], ascending=[False, False, True])
    keep_n = max(max_items - len(special_df), 0)
    selected = pd.concat([special_df, remain_df.head(keep_n)], ignore_index=True)
    selected = selected.drop_duplicates(subset=["path_str"], keep="first")
    return selected.sort_values(by=["scenario_count", "path_effect_abs", "path_str"], ascending=[False, False, True]).reset_index(drop=True)


def plot_chord(ax, retention_df: pd.DataFrame, future_only_df: pd.DataFrame, future_only_top_k: int = 10) -> None:
    apply_nature_style()
    top_future_only = future_only_df.head(future_only_top_k).copy()
    retained_agg = retention_df.groupby(["source_group", "future_channel"], as_index=False)["future_mean_abs"].sum().rename(columns={"future_mean_abs": "weight"})
    retained_agg["status"] = "retained"
    future_only_agg = top_future_only.groupby(["source_group", "future_channel"], as_index=False)["future_mean_abs"].sum().rename(columns={"future_mean_abs": "weight"})
    future_only_agg["status"] = "new"
    link_df = pd.concat([retained_agg, future_only_agg], ignore_index=True)
    link_df = link_df.loc[link_df["weight"] > 0].copy()

    source_weights = link_df.groupby("source_group")["weight"].sum().to_dict()
    channel_weights = link_df.groupby("future_channel")["weight"].sum().to_dict()
    left_spans = _weighted_spans(SOURCE_GROUP_ORDER, source_weights, 110.0, 250.0, 7.0)
    right_spans = _weighted_spans(CHANNEL_ORDER, channel_weights, -70.0, 70.0, 6.0)

    ax.set_facecolor(FIGURE_FACE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    outer_r = 0.46
    inner_r = 0.405
    for group_name in SOURCE_GROUP_ORDER:
        theta0, theta1, theta_mid = left_spans[group_name]
        ax.add_patch(Wedge((0.5, 0.5), r=outer_r, theta1=theta0, theta2=theta1, width=outer_r - inner_r, facecolor=GROUP_COLORS[group_name], edgecolor=FIGURE_FACE, linewidth=0.7, zorder=3))
        x_lab, y_lab = _polar_to_xy(theta_mid, outer_r + 0.07)
        ax.text(x_lab, y_lab, group_name, ha="center", va="center", color=GROUP_COLORS[group_name], fontweight="bold")

    channel_colors = {
        "Direct": "#CFC8C2",
        "via LAI": "#D7D0C9",
        "via UF": "#D4CEC8",
        "via WTD": "#DDD7D0",
        "Serial": "#E5DFD8",
    }
    for channel in CHANNEL_ORDER:
        theta0, theta1, theta_mid = right_spans[channel]
        ax.add_patch(Wedge((0.5, 0.5), r=outer_r, theta1=theta0, theta2=theta1, width=outer_r - inner_r, facecolor=channel_colors[channel], edgecolor=FIGURE_FACE, linewidth=0.7, zorder=3))
        x_lab, y_lab = _polar_to_xy(theta_mid, outer_r + 0.10)
        if channel == "Serial":
            ha = "center"
        else:
            ha = "left"
        ax.text(x_lab, y_lab, CHANNEL_LABEL_TEXT[channel], ha=ha, va="center", color=TEXT_SECONDARY, fontsize=8.0)

    max_weight = max(float(link_df["weight"].max()), 1e-9)
    for row in link_df.sort_values(by=["weight", "source_group", "future_channel"], ascending=[True, True, True]).itertuples(index=False):
        _, _, theta_left = left_spans[row.source_group]
        _, _, theta_right = right_spans[row.future_channel]
        start = _polar_to_xy(theta_left, inner_r - 0.01)
        end = _polar_to_xy(theta_right, inner_r - 0.01)
        width = 1.2 + 10.0 * float(row.weight) / max_weight
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-",
                linewidth=width,
                color=GROUP_COLORS[row.source_group],
                alpha=0.68 if row.status == "retained" else 0.25,
                linestyle="solid" if row.status == "retained" else (0, (2.2, 2.2)),
                connectionstyle="arc3,rad=0.18",
                zorder=2,
            )
        )

    ax.plot([0.19, 0.29], [0.09, 0.09], color=AXIS_COLOR, lw=2.2, solid_capstyle="round")
    ax.text(0.31, 0.09, "Retained paths", ha="left", va="center", color="#000000")
    ax.plot([0.58, 0.68], [0.09, 0.09], color=AXIS_COLOR, lw=2.2, linestyle=(0, (2.2, 2.2)), solid_capstyle="round")
    ax.text(0.70, 0.09, "Possible paths in the future", ha="left", va="center", color="#000000")


def plot_dumbbell(ax, impact_df: pd.DataFrame) -> None:
    apply_nature_style()
    plot_df = impact_df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df), dtype=float)

    ax.set_facecolor(FIGURE_FACE)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.85)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    max_x = max(float(plot_df["historical_abs"].max()), float(plot_df["future_abs"].max()), 1e-9)
    diff_sep_x = max_x * 1.05
    diff_col_x = max_x * 1.22
    for group_name in ["Hydrogeology", "Climate", "Anthropogenic"]:
        idxs = plot_df.index[plot_df["group_lv1"] == group_name].tolist()
        if idxs:
            ax.axhspan(min(idxs) - 0.45, max(idxs) + 0.45, facecolor=GROUP_COLORS[group_name], alpha=0.06, zorder=0)

    for idx, row in plot_df.iterrows():
        color = GROUP_COLORS[row["group_lv1"]]
        ax.hlines(idx, row["historical_abs"], row["future_abs"], color=color, linewidth=1.6, alpha=0.55, zorder=2)
        ax.scatter(row["historical_abs"], idx, s=28, color="white", edgecolors=color, linewidths=0.9, zorder=3)
        ax.scatter(
            row["future_abs"],
            idx,
            s=34,
            color=color,
            edgecolors=FIGURE_FACE,
            linewidths=0.7,
            alpha=0.92,
            zorder=4,
        )
        delta_text = f"{float(row['delta_abs']):+0.02f}"
        delta_color = SIGN_COLORS["positive"] if row["delta_sign"] == "positive" else SIGN_COLORS["negative"] if row["delta_sign"] == "negative" else TEXT_SECONDARY
        ax.text(diff_col_x, idx, delta_text, ha="center", va="center", color=delta_color)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["node"])
    ax.tick_params(axis="y", length=0, pad=1, labelsize=7.6)
    ax.set_xlim(0.0, max_x * 1.42)
    ax.set_xlabel("|Overall impact on the sinkhole|")
    ax.axvline(diff_sep_x, color=GRID_COLOR, linewidth=0.7, zorder=1)
    ax.text(diff_col_x, 1.02, "Difference", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=TEXT_SECONDARY)

    trans = ax.transAxes
    ax.scatter([0.05], [0.96], s=30, facecolors="white", edgecolors=AXIS_COLOR, linewidths=0.8, transform=trans, clip_on=False)
    ax.text(0.09, 0.96, "Historical", transform=trans, ha="left", va="center", color=TEXT_SECONDARY)
    ax.scatter([0.28], [0.96], s=34, color=AXIS_COLOR, edgecolors="none", transform=trans, clip_on=False)
    ax.text(0.32, 0.96, "Future", transform=trans, ha="left", va="center", color=TEXT_SECONDARY)


def plot_network(ax, current_edge_retention_df: pd.DataFrame, future_only_edge_df: pd.DataFrame, future_only_top_k: int = 12) -> None:
    apply_nature_style()
    ax.set_facecolor(FIGURE_FACE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    nodes = sorted(set(current_edge_retention_df["source"]).union(current_edge_retention_df["target"]).union(future_only_edge_df["source"]).union(future_only_edge_df["target"]), key=lambda n: (ROLE_X.get(n, 0.5), n))
    positions = _layout_nodes_rect(nodes)

    for x_val, header in COLUMN_HEADERS.items():
        if not any(abs(pos[0] - x_val) < 1e-9 for pos in positions.values()):
            continue
        fill = GROUP_COLORS["Anthropogenic"] if header == "Mediators" else GROUP_COLORS["Target"] if header == "Risk" else GROUP_COLORS["Climate"] if header == "Climate" else GROUP_COLORS["Hydrogeology"] if header == "Hydrogeology" else GROUP_COLORS["Anthropogenic"]
        ax.axvspan(max(0.0, x_val - 0.06), min(1.0, x_val + 0.06), ymin=0.06, ymax=0.93, facecolor=fill, alpha=0.06, zorder=0)
        ax.text(x_val, 0.96, header, ha="center", va="center", color=TEXT_SECONDARY)

    future_only_plot = future_only_edge_df.head(future_only_top_k).copy()
    future_only_max = max(float(future_only_plot["future_mean_abs"].max()), 1e-9) if len(future_only_plot) else 1.0
    for row in future_only_plot.sort_values(by=["future_mean_abs", "source", "target"], ascending=[True, True, True]).itertuples(index=False):
        source_xy = positions[row.source]
        target_xy = positions[row.target]
        width = 0.6 + 2.0 * float(row.future_mean_abs) / future_only_max
        ax.add_patch(
            FancyArrowPatch(
                posA=source_xy,
                posB=target_xy,
                arrowstyle="-|>",
                mutation_scale=6.5 + width,
                linewidth=width,
                color="#BEB7B0",
                alpha=0.45,
                linestyle=(0, (2.4, 2.4)),
                connectionstyle=f"arc3,rad={_edge_rad_rect(source_xy, target_xy)}",
                shrinkA=12.0,
                shrinkB=12.0,
                zorder=1,
            )
        )

    retained_df = current_edge_retention_df.loc[current_edge_retention_df["scenario_count"] > 0].copy()
    retained_max = max(float(retained_df["future_mean_abs"].max()), 1e-9)
    for row in retained_df.sort_values(by=["future_mean_abs", "source", "target"], ascending=[True, True, True]).itertuples(index=False):
        source_xy = positions[row.source]
        target_xy = positions[row.target]
        width = 0.8 + 3.1 * float(row.future_mean_abs) / retained_max
        ax.add_patch(
            FancyArrowPatch(
                posA=source_xy,
                posB=target_xy,
                arrowstyle="-|>",
                mutation_scale=7.0 + width,
                linewidth=width,
                color=SIGN_COLORS[row.future_sign],
                alpha=0.84 if row.retention_class == "Full" else 0.46,
                linestyle="solid" if row.retention_class == "Full" else (0, (2.2, 2.2)),
                connectionstyle=f"arc3,rad={_edge_rad_rect(source_xy, target_xy)}",
                shrinkA=12.0,
                shrinkB=12.0,
                zorder=2,
            )
        )

    node_w = 0.080
    node_h = 0.055
    for node in nodes:
        x_val, y_val = positions[node]
        ax.add_patch(
            FancyBboxPatch(
                (x_val - node_w / 2.0, y_val - node_h / 2.0),
                node_w,
                node_h,
                boxstyle="round,pad=0.01,rounding_size=0.008",
                linewidth=0.55,
                edgecolor=AXIS_COLOR,
                facecolor=GROUP_COLORS[_node_group(node)],
                zorder=3,
            )
        )
        ax.text(x_val, y_val, node, ha="center", va="center", zorder=4, fontweight="bold" if node == "RISK" else "normal")

    ax.plot([0.02, 0.08], [0.05, 0.05], color=SIGN_COLORS["positive"], lw=2.0, solid_capstyle="round")
    ax.text(0.09, 0.05, "Promotes disaster", ha="left", va="center", color="#000000")
    ax.plot([0.40, 0.46], [0.05, 0.05], color=SIGN_COLORS["negative"], lw=2.0, solid_capstyle="round")
    ax.text(0.47, 0.05, "Reduces disaster", ha="left", va="center", color="#000000")
    ax.plot([0.77, 0.83], [0.05, 0.05], color="#BEB7B0", lw=2.0, linestyle=(0, (2.4, 2.4)), solid_capstyle="round")
    ax.text(0.84, 0.05, "Possible paths in the future", ha="left", va="center", color="#000000")


def save_group2(out_path: str | Path = DEFAULT_OUT, summary_dir: str | Path = SUMMARY_DIR) -> Path:
    apply_nature_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(summary_dir)

    retention_df, future_only_df, current_edge_retention_df, future_only_edge_df, meta = build_retention_tables()
    impact_df = build_variable_impact_table()
    save_summary(summary_dir, retention_df, future_only_df, meta)

    fig = plt.figure(figsize=(7.35, 6.75), facecolor=FIGURE_FACE)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.00, 1.02],
        width_ratios=[1.00, 1.02],
        left=0.06,
        right=0.97,
        top=0.95,
        bottom=0.10,
        hspace=0.20,
        wspace=0.16,
    )
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])

    plot_chord(ax_tl, retention_df, future_only_df, future_only_top_k=10)
    plot_dumbbell(ax_tr, impact_df)
    plot_network(ax_b, current_edge_retention_df, future_only_edge_df, future_only_top_k=12)

    _add_panel_label(fig, ax_tl, "a")
    _add_panel_label(fig, ax_tr, "b")
    _add_panel_label(fig, ax_b, "c")

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final group2 figure.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = save_group2(args.out, args.summary_dir)
    print(f"[OK] group2 -> {out}")


if __name__ == "__main__":
    main()
