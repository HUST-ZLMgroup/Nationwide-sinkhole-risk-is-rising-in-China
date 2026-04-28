#!/usr/bin/env python3
"""Create a Morandi-style envelope-stacked bar chart for province climate-change risk variation."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


YEARS = ("2040", "2060", "2080", "2100")
BASE_YEAR = "2080"

YEAR_STYLES = {
    "2040": {"color": "#93A8B3", "label": "2040"},
    "2060": {"color": "#C8B79C", "label": "2060"},
    "2080": {"color": "#B78F91", "label": "2080"},
    "2100": {"color": "#A7B59F", "label": "2100"},
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Combine 2040/2060/2080/2100 province climate-change risk variation "
            "into a Morandi-style envelope-stacked horizontal bar chart."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="Project root containing outputs.")
    parser.add_argument("--path-name", default="Points_China_all_10km", help="Point dataset folder under outputs.")
    parser.add_argument("--ssp", default="ssp2", help="SSP scenario folder.")
    parser.add_argument("--years", nargs="+", default=list(YEARS), help="Years to combine.")
    parser.add_argument("--base-year", default=BASE_YEAR, help="Year used for province sorting.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to outputs/<path>/<ssp>/combined_climate_change.")
    parser.add_argument("--output-stem", default=None, help="Output filename without suffix.")
    return parser.parse_args()


def find_summary_csv(repo_root: Path, path_name: str, ssp: str, year: str) -> Path:
    climate_dir = repo_root / "outputs" / path_name / ssp / year / "climate_change"
    matches = sorted(climate_dir.glob(f"province_climate_change_summary_{path_name}_{ssp}_{year}.csv"))
    if not matches:
        matches = sorted(climate_dir.glob("province_climate_change_summary_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No province summary CSV found under {climate_dir}")
    return matches[0]


def find_lollipop_svg(repo_root: Path, path_name: str, ssp: str, year: str) -> Path | None:
    climate_dir = repo_root / "outputs" / path_name / ssp / year / "climate_change"
    matches = sorted(climate_dir.glob(f"province_change_lollipop_raw_gwr_{path_name}_{ssp}_{year}.svg"))
    if not matches:
        matches = sorted(climate_dir.glob("*lollipop*.svg"))
    return matches[0] if matches else None


def read_year_values(csv_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "NAME_EN_JX" not in (reader.fieldnames or []):
            raise KeyError(f"{csv_path} is missing NAME_EN_JX")
        for row in reader:
            name = (row.get("NAME_EN_JX") or "").strip()
            if not name:
                continue
            raw = row.get("change_rate_pct")
            multiplier = 1.0
            if raw in (None, ""):
                raw = row.get("change_rate")
                multiplier = 100.0
            if raw in (None, ""):
                continue
            try:
                values[name] = float(raw) * multiplier
            except ValueError:
                continue
    return values


def ordered_provinces(data_by_year: dict[str, dict[str, float]], base_year: str) -> list[str]:
    base_values = data_by_year[base_year]
    ordered = sorted(base_values, key=lambda name: base_values[name], reverse=True)

    all_names = set().union(*(set(v) for v in data_by_year.values()))
    extras = sorted(
        all_names - set(ordered),
        key=lambda name: max(
            (values[name] for values in data_by_year.values() if name in values),
            default=-math.inf,
        ),
        reverse=True,
    )
    return ordered + extras


def envelope_limits(provinces: list[str], years: list[str], data_by_year: dict[str, dict[str, float]]) -> tuple[float, float]:
    lower_endpoint = 0.0
    upper_endpoint = 0.0
    for province in provinces:
        for year in years:
            value = data_by_year[year].get(province)
            if value is None or not math.isfinite(value):
                continue
            lower_endpoint = min(lower_endpoint, value)
            upper_endpoint = max(upper_endpoint, value)

    span = max(upper_endpoint - lower_endpoint, 1.0)
    pad = span * 0.08
    lower = math.floor((lower_endpoint - pad) / 10.0) * 10.0
    upper = math.ceil((upper_endpoint + pad) / 10.0) * 10.0
    return lower, upper


def envelope_segments(
    province: str,
    years: list[str],
    data_by_year: dict[str, dict[str, float]],
    positive: bool,
) -> list[tuple[str, float, float]]:
    year_order = {year: idx for idx, year in enumerate(years)}
    entries: list[tuple[float, int, str, float]] = []
    for year in years:
        value = data_by_year[year].get(province)
        if value is None or not math.isfinite(value):
            continue
        if positive and value > 0:
            entries.append((abs(value), year_order[year], year, value))
        elif not positive and value < 0:
            entries.append((abs(value), year_order[year], year, value))

    entries.sort()
    segments: list[tuple[str, float, float]] = []
    previous = 0.0
    for _, _, year, value in entries:
        width = value - previous
        if abs(width) > 1e-12:
            segments.append((year, previous, width))
        previous = value
    return segments


def tick_values(xlim: tuple[float, float]) -> list[float]:
    lower, upper = xlim
    span = upper - lower
    if span <= 120:
        step = 20
    elif span <= 220:
        step = 40
    else:
        step = 50
    start = math.ceil(lower / step) * step
    stop = math.floor(upper / step) * step
    ticks = []
    value = start
    while value <= stop + 1e-9:
        ticks.append(value)
        value += step
    if 0 not in ticks and lower < 0 < upper:
        ticks.append(0)
        ticks.sort()
    return ticks


def write_combined_csv(output_csv: Path, provinces: list[str], years: list[str], data_by_year: dict[str, dict[str, float]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["NAME_EN_JX", *years])
        writer.writeheader()
        for province in provinces:
            row = {"NAME_EN_JX": province}
            for year in years:
                value = data_by_year.get(year, {}).get(province)
                row[year] = "" if value is None else f"{value:.10g}"
            writer.writerow(row)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 31,
            "axes.linewidth": 1.8,
            "axes.edgecolor": "#8a8a8a",
            "xtick.major.width": 1.6,
            "ytick.major.width": 0.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_combined(
    provinces: list[str],
    years: list[str],
    data_by_year: dict[str, dict[str, float]],
    output_stem: Path,
) -> None:
    configure_matplotlib()

    n = len(provinces)
    y_positions = list(range(n))
    font_size = 31
    fig_height = max(18.0, 0.50 * n)
    fig_width = fig_height * 9.0 / 16.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    xlim = envelope_limits(provinces, years, data_by_year)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=(0, (5, 5)), color="#7c7c7c", linewidth=0.8, alpha=0.95)
    ax.yaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#8a8a8a")
        spine.set_linewidth(1.8)

    bar_height = 0.74
    for y, province in zip(y_positions, provinces):
        province_segments = [
            *envelope_segments(province, years, data_by_year, positive=False),
            *envelope_segments(province, years, data_by_year, positive=True),
        ]
        for year, left, width in province_segments:
            style = YEAR_STYLES.get(year, {"color": "#555555", "label": year})
            ax.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=style["color"],
                alpha=0.86,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

    if "China" in provinces:
        china_idx = provinces.index("China")
        ax.axhspan(
            china_idx - 0.38,
            china_idx + 0.38,
            facecolor="none",
            edgecolor="#ec8e9c",
            linewidth=2.2,
            zorder=7,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    ax.set_xticks(tick_values(xlim))
    ax.axvline(0, color="#0a6ea8", linewidth=3.6, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(provinces, rotation=0, ha="right", fontsize=font_size)
    for label in ax.get_yticklabels():
        if label.get_text().strip() == "China":
            label.set_color("#d96d80")
            label.set_fontweight("bold")

    ax.tick_params(axis="x", width=1.6, length=8, color="#8a8a8a", labelsize=font_size)
    ax.tick_params(axis="y", width=0, length=0, color="#8a8a8a", labelsize=font_size)

    handles = [
        Patch(
            facecolor=YEAR_STYLES.get(year, {"color": "#555555"})["color"],
            edgecolor="none",
            alpha=0.86,
            label=year,
        )
        for year in years
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.985),
        ncol=1,
        frameon=False,
        fontsize=font_size,
        handlelength=1.25,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.0,
    )

    fig.text(0.58, 0.02, "Risk variation under climate change (%)", ha="center", fontsize=font_size)
    fig.subplots_adjust(left=0.34, right=0.985, bottom=0.10, top=0.985)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {"format": "svg"},
        ".pdf": {"format": "pdf"},
        ".png": {"format": "png", "dpi": 600},
    }.items():
        fig.savefig(
            output_stem.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=1 / 25.4,
            facecolor="white",
            edgecolor="none",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "outputs" / args.path_name / args.ssp / "combined_climate_change"
    )
    output_stem = (
        output_dir / args.output_stem
        if args.output_stem
        else output_dir / f"province_change_envelope_stacked_bar_morandi_{args.path_name}_{args.ssp}_{'_'.join(args.years)}"
    )

    data_by_year: dict[str, dict[str, float]] = {}
    print("Input province summaries:")
    for year in args.years:
        csv_path = find_summary_csv(repo_root, args.path_name, args.ssp, year)
        svg_path = find_lollipop_svg(repo_root, args.path_name, args.ssp, year)
        data_by_year[year] = read_year_values(csv_path)
        print(f"  {year}: {csv_path}")
        if svg_path:
            print(f"        existing lollipop: {svg_path}")

    if args.base_year not in data_by_year:
        raise KeyError(f"Base year {args.base_year} is not in {args.years}")

    provinces = ordered_provinces(data_by_year, args.base_year)
    combined_csv = output_stem.with_suffix(".csv")
    write_combined_csv(combined_csv, provinces, args.years, data_by_year)
    plot_combined(provinces, args.years, data_by_year, output_stem)

    print(f"Combined data saved: {combined_csv}")
    print(f"Figure saved: {output_stem.with_suffix('.svg')}")
    print(f"Figure saved: {output_stem.with_suffix('.pdf')}")
    print(f"Figure saved: {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    main()
