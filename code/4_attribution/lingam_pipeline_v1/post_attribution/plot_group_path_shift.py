from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from lingam_pipeline_v1.pre_attribution.attribution_config import FIGURE_FACE, GRID_COLOR, GROUP_COLORS, OUTPUT_ROOT, TEXT_SECONDARY
from lingam_pipeline_v1.post_attribution.nature_figure_style import apply_nature_style


DEFAULT_GROUP_CONTRIBUTION_CSV = OUTPUT_ROOT / "future" / "summary" / "group_path_contribution_summary.csv"


def load_group_path_contributions(contribution_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(contribution_csv)


def _coarse_group(path_group: str) -> str:
    if str(path_group).startswith("Hydrogeology"):
        return "Hydrogeology"
    if str(path_group).startswith("Climate"):
        return "Climate"
    if str(path_group).startswith("Anthropogenic"):
        return "Anthropogenic"
    return str(path_group)


def available_years(contribution_df: pd.DataFrame) -> list[int]:
    return sorted(pd.Series(contribution_df["year"]).dropna().astype(int).unique().tolist())


def _scenario_group_contributions(contribution_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    plot_df = contribution_df.copy()
    if value_col not in plot_df.columns:
        raise KeyError(f"Missing required column in contribution table: {value_col}")
    plot_df["coarse_group"] = plot_df["path_group"].map(_coarse_group)
    return (
        plot_df.groupby(["scenario_id", "ssp", "year", "coarse_group"], as_index=False)[value_col]
        .sum()
        .sort_values(by=["year", "ssp", "coarse_group"])
    )


def plot_group_path_shift(
    ax: Axes,
    contribution_df: pd.DataFrame,
    year: int,
    value_col: str = "contribution_share",
    show_scale: bool = False,
) -> Axes:
    apply_nature_style()
    plot_df = _scenario_group_contributions(contribution_df, value_col=value_col)
    year_df = plot_df.loc[plot_df["year"].astype(int) == int(year)].copy()
    if year_df.empty:
        raise ValueError(f"No contribution records available for year={year}")

    theta_map = {
        "Climate": 0.0,
        "Hydrogeology": 2.0 * np.pi / 3.0,
        "Anthropogenic": 4.0 * np.pi / 3.0,
    }
    group_order = ["Climate", "Hydrogeology", "Anthropogenic"]
    ssp_order = sorted(year_df["ssp"].drop_duplicates().tolist())
    petal_width = 2.0 * np.pi / 3.0 * 0.38
    offsets = np.linspace(-petal_width * 0.34, petal_width * 0.34, num=len(ssp_order))

    ax.set_facecolor(FIGURE_FACE)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 0.68)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    theta_ref = np.linspace(0.0, 2.0 * np.pi, 360)
    for radius in (0.2, 0.4, 0.6):
        ax.plot(theta_ref, np.full_like(theta_ref, radius), color=GRID_COLOR, linewidth=0.5, zorder=0)
    for theta in theta_map.values():
        ax.plot([theta, theta], [0.0, 0.62], color=GRID_COLOR, linewidth=0.5, zorder=0)

    for group_name in group_order:
        group_df = year_df.loc[year_df["coarse_group"] == group_name].sort_values("ssp")
        values = group_df[value_col].to_numpy(dtype=float)
        for offset, value in zip(offsets, values):
            ax.bar(
                theta_map[group_name] + offset,
                value,
                width=petal_width / 4.6,
                bottom=0.0,
                color=GROUP_COLORS[group_name],
                alpha=0.18,
                edgecolor="none",
                zorder=1,
            )
        mean_value = float(values.mean()) if len(values) else 0.0
        ax.bar(
            theta_map[group_name],
            mean_value,
            width=petal_width,
            bottom=0.0,
            color=GROUP_COLORS[group_name],
            alpha=0.88,
            edgecolor=FIGURE_FACE,
            linewidth=0.7,
            zorder=2,
        )
        ax.scatter(
            [theta_map[group_name]],
            [mean_value],
            s=20,
            color="white",
            edgecolors=GROUP_COLORS[group_name],
            linewidths=0.7,
            zorder=3,
        )
        ax.text(theta_map[group_name], 0.665, group_name, ha="center", va="center", color=GROUP_COLORS[group_name])

    center_disc = Circle(
        (0.5, 0.5),
        radius=0.12,
        transform=ax.transAxes,
        facecolor=FIGURE_FACE,
        edgecolor=TEXT_SECONDARY,
        linewidth=0.6,
        zorder=5,
    )
    ax.add_patch(center_disc)
    ax.text(0.5, 0.5, str(year), transform=ax.transAxes, ha="center", va="center", fontweight="bold", zorder=6)

    if show_scale:
        ax.text(np.deg2rad(18.0), 0.20, "0.2", ha="left", va="center", color=TEXT_SECONDARY)
        ax.text(np.deg2rad(18.0), 0.40, "0.4", ha="left", va="center", color=TEXT_SECONDARY)
        ax.text(np.deg2rad(18.0), 0.60, "0.6", ha="left", va="center", color=TEXT_SECONDARY)
        ax.text(
            0.02,
            0.03,
            "Wide petals show mean share;\nthin petals show SSP-specific values",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=TEXT_SECONDARY,
        )

    return ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot small-multiple contribution petals by year.")
    parser.add_argument("--contribution-csv", type=Path, default=DEFAULT_GROUP_CONTRIBUTION_CSV)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_group_path_contributions(args.contribution_csv)
    years = available_years(df)
    fig = plt.figure(figsize=(5.8, 5.6), facecolor=FIGURE_FACE)
    gs = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.12)
    for idx, year in enumerate(years[:4]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection="polar")
        plot_group_path_shift(ax, df, year=year, show_scale=(idx == 0))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, format="svg", bbox_inches="tight")
        print(f"[OK] group_path_shift.svg -> {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
