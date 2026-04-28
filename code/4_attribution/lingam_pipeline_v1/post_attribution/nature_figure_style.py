from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    AXIS_COLOR,
    BASE_FONT_SIZE,
    COMMON_OUTPUT_DIR,
    FIGURE_FACE,
    FONT_FAMILY,
    FONT_SERIF_FALLBACKS,
    GRID_COLOR,
    GROUP_COLORS,
    SEQUENTIAL_CMAP_HEX,
    SIGN_COLORS,
    TEXT_SECONDARY,
    ensure_common_output_dir,
)


FONT_FAMILY_SERIF = FONT_SERIF_FALLBACKS


def apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(FONT_FAMILY_SERIF),
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE,
            "figure.facecolor": FIGURE_FACE,
            "axes.facecolor": FIGURE_FACE,
            "savefig.facecolor": FIGURE_FACE,
            "savefig.edgecolor": FIGURE_FACE,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )


def morandi_sequential_cmap(name: str = "morandi_seq") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, list(SEQUENTIAL_CMAP_HEX))


def save_palette_demo(out_path: str | Path | None = None) -> Path:
    apply_nature_style()
    out_dir = ensure_common_output_dir(COMMON_OUTPUT_DIR if out_path is None else Path(out_path).parent)
    path = Path(out_path) if out_path is not None else out_dir / "nature_style_palette_demo.svg"

    fig = plt.figure(figsize=(7.2, 2.4), facecolor=FIGURE_FACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.45)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    ax_top.set_xlim(0, 4)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    labels = ["Hydrogeology", "Climate", "Anthropogenic", "Target / Risk"]
    colors = [
        GROUP_COLORS["Hydrogeology"],
        GROUP_COLORS["Climate"],
        GROUP_COLORS["Anthropogenic"],
        GROUP_COLORS["Target"],
    ]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax_top.add_patch(Rectangle((i + 0.10, 0.20), 0.55, 0.42, facecolor=color, edgecolor=AXIS_COLOR, linewidth=0.6))
        ax_top.text(i + 0.78, 0.41, label, va="center", ha="left")

    cmap = morandi_sequential_cmap()
    gradient = [list(range(256))]
    ax_bot.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0.20, 0.62])
    ax_bot.add_patch(Rectangle((0, 0.20), 1, 0.42, fill=False, edgecolor=AXIS_COLOR, linewidth=0.6))
    ax_bot.text(0, 0.88, "Sequential heatmap palette", ha="left", va="bottom")
    ax_bot.text(0, 0.02, "0.0", ha="left", va="bottom", color=TEXT_SECONDARY)
    ax_bot.text(1, 0.02, "1.0", ha="right", va="bottom", color=TEXT_SECONDARY)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis("off")

    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a palette demo for the shared Nature figure style.")
    parser.add_argument(
        "--out",
        type=Path,
        default=COMMON_OUTPUT_DIR / "nature_style_palette_demo.svg",
        help="Output SVG path for the palette demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = save_palette_demo(args.out)
    print(f"[OK] nature_style_palette_demo.svg -> {path}")
    print(f"[INFO] Primary font family configured as: {FONT_FAMILY}")
    print(f"[INFO] Positive/negative sign colors: {SIGN_COLORS}")


if __name__ == "__main__":
    main()
