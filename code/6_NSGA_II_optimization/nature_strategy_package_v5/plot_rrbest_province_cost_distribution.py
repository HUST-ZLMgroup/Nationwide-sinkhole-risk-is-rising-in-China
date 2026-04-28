from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ISA_COLOR = "#E2CDAF"
WTD_COLOR = "#D08A64"
LAI_COLOR = "#8E6D7B"
TOTAL_LINE_COLOR = "#7A645B"
CHINA_EDGE = "#6E5A53"
GRID_COLOR = "#DDD4CD"
TEXT_COLOR = "#2F2F2F"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#4A4A4A",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": TEXT_COLOR,
    }
)


DOMINANT_ORDER = {"Cost_ISA": 0, "Cost_WT": 1, "Cost_LAI": 2}
DOMINANT_LABEL = {
    "Cost_ISA": "ISA-dominant",
    "Cost_WT": "WTD-dominant",
    "Cost_LAI": "LAI-dominant",
}
FIG_SIZE = (18.0, 6.0)
FONT_SCALE = 1.2328
ABSOLUTE_FONT_MULTIPLIER = 1.9565


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True, help="Path to nsga2_*_summary.csv")
    parser.add_argument("--out-dir", required=True, help="Base output directory")
    parser.add_argument(
        "--tag",
        default="province_cost_distribution_rrbest",
        help="Subfolder name to create under out-dir",
    )
    return parser.parse_args()


def build_distribution(summary_df: pd.DataFrame) -> pd.DataFrame:
    required = {"NAME_EN_JX", "Cost_ISA", "Cost_WT", "Cost_LAI", "sample_weight"}
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def aggregate_region(region_name: str, sub: pd.DataFrame) -> dict:
        weights = sub["sample_weight"].to_numpy(dtype=float)
        represented_n = float(np.sum(weights))
        represented_area_km2 = represented_n * 100.0

        cost_isa_total = float(np.sum(sub["Cost_ISA"].to_numpy(dtype=float) * weights))
        cost_wtd_total = float(np.sum(sub["Cost_WT"].to_numpy(dtype=float) * weights))
        cost_lai_total = float(np.sum(sub["Cost_LAI"].to_numpy(dtype=float) * weights))
        cost_total = cost_isa_total + cost_wtd_total + cost_lai_total

        mean_cost_isa = cost_isa_total / max(represented_n, 1e-12)
        mean_cost_wtd = cost_wtd_total / max(represented_n, 1e-12)
        mean_cost_lai = cost_lai_total / max(represented_n, 1e-12)
        mean_cost_total = mean_cost_isa + mean_cost_wtd + mean_cost_lai

        return {
            "region": region_name,
            "represented_n_100km2": represented_n,
            "represented_area_km2": represented_area_km2,
            "weighted_total_cost_ISA_usd": cost_isa_total,
            "weighted_total_cost_WTD_usd": cost_wtd_total,
            "weighted_total_cost_LAI_usd": cost_lai_total,
            "weighted_total_cost_usd": cost_total,
            "mean_cost_ISA_usd_per_100km2": mean_cost_isa,
            "mean_cost_WTD_usd_per_100km2": mean_cost_wtd,
            "mean_cost_LAI_usd_per_100km2": mean_cost_lai,
            "mean_cost_total_usd_per_100km2": mean_cost_total,
        }

    records = [aggregate_region("China", summary_df)]
    for region_name, sub in summary_df.groupby("NAME_EN_JX", dropna=True):
        records.append(aggregate_region(str(region_name), sub))
    data = pd.DataFrame.from_records(records)

    data["ISA_share"] = data["mean_cost_ISA_usd_per_100km2"] / np.maximum(data["mean_cost_total_usd_per_100km2"], 1e-12)
    data["WTD_share"] = data["mean_cost_WTD_usd_per_100km2"] / np.maximum(data["mean_cost_total_usd_per_100km2"], 1e-12)
    data["LAI_share"] = data["mean_cost_LAI_usd_per_100km2"] / np.maximum(data["mean_cost_total_usd_per_100km2"], 1e-12)
    data["mean_cost_total_million_usd_per_100km2"] = data["mean_cost_total_usd_per_100km2"] / 1e6
    data["mean_cost_ISA_million_usd_per_100km2"] = data["mean_cost_ISA_usd_per_100km2"] / 1e6
    data["mean_cost_WTD_million_usd_per_100km2"] = data["mean_cost_WTD_usd_per_100km2"] / 1e6
    data["mean_cost_LAI_million_usd_per_100km2"] = data["mean_cost_LAI_usd_per_100km2"] / 1e6
    data["mean_cost_total_usd_per_km2"] = data["mean_cost_total_usd_per_100km2"] / 100.0
    data["mean_cost_ISA_usd_per_km2"] = data["mean_cost_ISA_usd_per_100km2"] / 100.0
    data["mean_cost_WTD_usd_per_km2"] = data["mean_cost_WTD_usd_per_100km2"] / 100.0
    data["mean_cost_LAI_usd_per_km2"] = data["mean_cost_LAI_usd_per_100km2"] / 100.0
    data["mean_cost_total_million_usd_per_km2"] = data["mean_cost_total_usd_per_km2"] / 1e6
    data["mean_cost_ISA_million_usd_per_km2"] = data["mean_cost_ISA_usd_per_km2"] / 1e6
    data["mean_cost_WTD_million_usd_per_km2"] = data["mean_cost_WTD_usd_per_km2"] / 1e6
    data["mean_cost_LAI_million_usd_per_km2"] = data["mean_cost_LAI_usd_per_km2"] / 1e6
    data["dominant_component"] = data[
        [
            "mean_cost_ISA_usd_per_100km2",
            "mean_cost_WTD_usd_per_100km2",
            "mean_cost_LAI_usd_per_100km2",
        ]
    ].idxmax(axis=1)
    data["dominant_component"] = data["dominant_component"].map(
        {
            "mean_cost_ISA_usd_per_100km2": "Cost_ISA",
            "mean_cost_WTD_usd_per_100km2": "Cost_WT",
            "mean_cost_LAI_usd_per_100km2": "Cost_LAI",
        }
    )
    data["dominant_share"] = data[["ISA_share", "WTD_share", "LAI_share"]].max(axis=1)

    data = data.sort_values(
        by=["mean_cost_total_million_usd_per_km2", "region"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return data


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def get_group_breaks(plot_df: pd.DataFrame) -> list[tuple[str, int, int]]:
    province_df = plot_df[plot_df["region"] != "China"].copy().reset_index(drop=True)
    if province_df.empty:
        return []

    group_breaks: list[tuple[str, int, int]] = []
    start_idx = 0
    dom_values = province_df["dominant_component"].tolist()
    for i in range(1, len(dom_values) + 1):
        if i == len(dom_values) or dom_values[i] != dom_values[start_idx]:
            group_breaks.append((dom_values[start_idx], start_idx + 1, i))
            start_idx = i
    return group_breaks


def style_xticklabels(ax, font_scale: float = FONT_SCALE):
    ax.tick_params(axis="x", pad=2)
    for label in ax.get_xticklabels():
        label.set_rotation(58)
        label.set_ha("right")
        label.set_va("top")
        label.set_rotation_mode("anchor")
        label.set_fontsize(8.8 * font_scale)


def add_group_guides(ax, plot_df: pd.DataFrame, y_top: float):
    _ = (ax, plot_df, y_top)
    return


def make_percentage_figure(plot_df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")
    style_axis(ax)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.44, top=0.87)

    x = np.arange(len(plot_df))
    china_idx = int(plot_df.index[plot_df["region"] == "China"][0])
    isa = plot_df["ISA_share"].to_numpy(dtype=float) * 100.0
    wtd = plot_df["WTD_share"].to_numpy(dtype=float) * 100.0
    lai = plot_df["LAI_share"].to_numpy(dtype=float) * 100.0
    regions = plot_df["region"].tolist()

    widths = np.full(len(plot_df), 0.72, dtype=float)
    widths[0] = 0.82

    ax.bar(x, isa, color=ISA_COLOR, edgecolor="none", width=widths, label="ISA")
    ax.bar(x, wtd, bottom=isa, color=WTD_COLOR, edgecolor="none", width=widths, label="WTD")
    ax.bar(x, lai, bottom=isa + wtd, color=LAI_COLOR, edgecolor="none", width=widths, label="LAI")

    ax.bar(
        x[china_idx],
        100.0,
        bottom=0.0,
        width=widths[china_idx],
        facecolor="none",
        edgecolor=CHINA_EDGE,
        linewidth=1.3,
    )

    ax.set_xlim(-0.6, len(plot_df) - 0.4)
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{v:d}%" for v in range(0, 101, 20)], fontsize=10 * FONT_SCALE)
    ax.set_ylabel("Cost share within the RR-best strategy", fontsize=11.8 * FONT_SCALE)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    style_xticklabels(ax, FONT_SCALE)
    ax.get_xticklabels()[china_idx].set_fontweight("bold")

    legend = ax.legend(
        frameon=False,
        ncol=3,
        fontsize=10 * FONT_SCALE,
        handlelength=1.6,
        columnspacing=1.4,
        loc="upper right",
        bbox_to_anchor=(0.996, 1.113),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_absolute_figure(plot_df: pd.DataFrame, save_path: Path):
    abs_font_scale = FONT_SCALE * ABSOLUTE_FONT_MULTIPLIER
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")
    style_axis(ax)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.44, top=0.87)

    x = np.arange(len(plot_df))
    china_idx = int(plot_df.index[plot_df["region"] == "China"][0])
    isa = plot_df["mean_cost_ISA_million_usd_per_km2"].to_numpy(dtype=float)
    wtd = plot_df["mean_cost_WTD_million_usd_per_km2"].to_numpy(dtype=float)
    lai = plot_df["mean_cost_LAI_million_usd_per_km2"].to_numpy(dtype=float)
    total = plot_df["mean_cost_total_million_usd_per_km2"].to_numpy(dtype=float)
    regions = plot_df["region"].tolist()

    widths = np.full(len(plot_df), 0.72, dtype=float)
    widths[0] = 0.82

    ax.bar(x, isa, color=ISA_COLOR, edgecolor="none", width=widths, label="ISA")
    ax.bar(x, wtd, bottom=isa, color=WTD_COLOR, edgecolor="none", width=widths, label="WTD")
    ax.bar(x, lai, bottom=isa + wtd, color=LAI_COLOR, edgecolor="none", width=widths, label="LAI")
    ax.plot(
        x,
        total,
        color=TOTAL_LINE_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=6.6,
        markerfacecolor="white",
        markeredgecolor=TOTAL_LINE_COLOR,
        markeredgewidth=1.4,
        zorder=5,
        label="Total",
    )

    ax.bar(
        x[china_idx],
        total[china_idx],
        bottom=0.0,
        width=widths[china_idx],
        facecolor="none",
        edgecolor=CHINA_EDGE,
        linewidth=1.3,
    )
    y_max = float(np.max(total))

    ax.set_xlim(-0.6, len(plot_df) - 0.4)
    ax.set_ylim(0, y_max * 1.10)
    ax.tick_params(axis="y", labelsize=10 * abs_font_scale)
    ax.set_ylabel("Mean cost per km² (million USD)", fontsize=11.8 * abs_font_scale)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    style_xticklabels(ax, abs_font_scale)
    ax.get_xticklabels()[china_idx].set_fontweight("bold")

    legend = ax.legend(
        frameon=False,
        ncol=4,
        fontsize=10 * abs_font_scale,
        handlelength=1.6,
        columnspacing=1.4,
        loc="upper right",
        bbox_to_anchor=(0.996, 1.09),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    ax.text(
        x[china_idx],
        total[china_idx] + y_max * 0.015,
        f"{total[china_idx]:.3f}",
        fontsize=8.9 * abs_font_scale,
        ha="center",
        va="bottom",
        color=CHINA_EDGE,
        fontweight="bold",
    )

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve() / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv)
    plot_df = build_distribution(summary_df)

    csv_path = out_dir / "province_cost_distribution_rrbest.csv"
    share_fig_path = out_dir / "province_cost_share_rrbest.svg"
    absolute_fig_path = out_dir / "province_cost_absolute_rrbest.svg"
    plot_df.to_csv(csv_path, index=False)
    make_percentage_figure(plot_df, share_fig_path)
    make_absolute_figure(plot_df, absolute_fig_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved percentage figure: {share_fig_path}")
    print(f"Saved absolute figure: {absolute_fig_path}")


if __name__ == "__main__":
    main()
