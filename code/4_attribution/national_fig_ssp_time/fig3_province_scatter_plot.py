import os
import re
import glob
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Silence Matplotlib font fallback chatter (keeps notebook output clean)
mpl.set_loglevel("error")
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# --- Colors / style (aligned with the notebook figures) ---
SSP_COLOR = {
    "ssp1": "#8694b8",  # green
    "ssp2": "#9570b3",  # olive/green
    "ssp3": "#FB7D5D",  # purple
    "ssp4": "#b44f02",  # yellow/orange
    "ssp5": "#DC3739AA",  # orange/red
}

SPINE_GRAY = "#7A7A7A"
TEXT_BLACK = "black"

# Global typography requested by the user
# Use Times New Roman when available; fall back to other serif fonts gracefully.
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9,
        "text.color": TEXT_BLACK,
        "axes.labelcolor": TEXT_BLACK,
        "xtick.color": TEXT_BLACK,
        "ytick.color": TEXT_BLACK,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

TNR = FontProperties(family=["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"])


def infer_ssp_and_year_from_filename(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer ssp (ssp1..ssp5) and year (e.g. 2100) from the filename."""
    base = os.path.basename(path)
    m = re.search(r"_((?:ssp)[1-5])_(\d{4})\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2)


def load_province_change_csv(path: str) -> pd.DataFrame:
    """Load the province change CSV (headered)."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalize possible BOM-ed header
    if "\ufeffADCODE99" in df.columns and "ADCODE99" not in df.columns:
        df = df.rename(columns={"\ufeffADCODE99": "ADCODE99"})

    numeric_cols = [
        "n_points",
        "mean_prob_2020",
        "mean_prob_current",
        "delta_mean",
        "change_rate",
        "change_rate_pct",
        "median_prob_2020",
        "median_prob_current",
        "delta_median",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def apply_axes_style(ax: plt.Axes) -> None:
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE_GRAY)
    ax.tick_params(axis="both", labelcolor=TEXT_BLACK, color=SPINE_GRAY)
    for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
        lab.set_fontproperties(TNR)
        lab.set_color(TEXT_BLACK)


def make_fig3_scatter_only(
    df: pd.DataFrame,
    ssp: str,
    year: str,
    save_path: str,
    width_cm: float = 6.0,
    height_cm: float = 6.0,
) -> str:
    """Create and save Figure 3 (right panel only): baseline vs future scatter."""
    ssp = (ssp or "ssp5").lower()
    color = SSP_COLOR.get(ssp, "#FB7D5D")

    # Figure size requested by the user (6cm x 6cm)
    fig_w_in = float(width_cm) / 2.54
    fig_h_in = float(height_cm) / 2.54

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    required = ["mean_prob_2020", "mean_prob_current", "delta_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    d = df.dropna(subset=required).copy()
    x = d["mean_prob_2020"].to_numpy(dtype=float)
    y = d["mean_prob_current"].to_numpy(dtype=float)

    if "n_points" in d.columns:
        npts = d["n_points"].to_numpy(dtype=float)
        npts = np.nan_to_num(npts, nan=0.0, posinf=0.0, neginf=0.0)
        maxn = float(max(npts.max(), 1.0))
        # small figure: keep marker sizes compact
        sizes = 12.0 + 36.0 * np.sqrt(npts / maxn)
    else:
        sizes = np.full_like(x, 18.0, dtype=float)

    neg = d["delta_mean"].to_numpy(dtype=float) < 0

    # Positive / non-negative change
    ax.scatter(
        x[~neg],
        y[~neg],
        s=sizes[~neg],
        color=color,
        alpha=0.65,
        linewidth=0,
        zorder=2,
    )

    # Negative change (hollow markers)
    ax.scatter(
        x[neg],
        y[neg],
        s=sizes[neg],
        facecolors="none",
        edgecolors=SPINE_GRAY,
        linewidth=0.8,
        zorder=3,
    )

    # y=x reference line
    finite = np.isfinite(x) & np.isfinite(y)
    lim_max = float(max(np.nanmax(x[finite]), np.nanmax(y[finite])) * 1.05) if finite.any() else 0.1
    lim_max = max(lim_max, 0.1)
    ax.plot([0, lim_max], [0, lim_max], linestyle="--", color=SPINE_GRAY, linewidth=0.8, zorder=1)

    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    ax.set_xlabel("Mean probability (2020)", fontproperties=TNR, color=TEXT_BLACK)
    ax.set_ylabel(f"Mean probability ({year})", fontproperties=TNR, color=TEXT_BLACK)

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    ax.grid(True, axis="both", linewidth=0.5, alpha=0.25)

    # Small in-plot label
    ax.text(
        0.02,
        0.98,
        f"{ssp.upper()} {year}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontproperties=TNR,
        fontsize=9,
        color=TEXT_BLACK,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=5,
            label="Δmean ≥ 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=SPINE_GRAY,
            markersize=5,
            label="Δmean < 0",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        prop=TNR,
        fontsize=9,
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

    apply_axes_style(ax)

    # Keep the canvas size exactly 6cm x 6cm; control margins manually
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def resolve_input_path(in_path: Optional[str] = None, out_dir: Optional[str] = None) -> str:
    if in_path and os.path.exists(in_path):
        return in_path

    # Try notebook variables if running via %run
    g = globals()
    for key in ["in_path", "IN_PATH"]:
        p = g.get(key)
        if isinstance(p, str) and os.path.exists(p):
            return p

    search_dir = out_dir or g.get("out_dir") or "."
    candidates = glob.glob(
        os.path.join(search_dir, "**", "province_prob_change_vs2020_lonlat_align_clip01_sortByChangeRate_*_*.csv"),
        recursive=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No matching province_prob_change_vs2020...csv found. "
            "Please set in_path (full file path) or ensure files are under out_dir."
        )
    return candidates[0]


def main(
    in_path: Optional[str] = None,
    out_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    ssp: Optional[str] = None,
    year: Optional[str] = None,
    width_cm: float = 6.0,
    height_cm: float = 6.0,
) -> str:
    in_path = resolve_input_path(in_path, out_dir)
    ssp_infer, year_infer = infer_ssp_and_year_from_filename(in_path)
    ssp = (ssp or ssp_infer or "ssp5").lower()
    year = year or year_infer or "2100"

    df = load_province_change_csv(in_path)

    if out_path is None:
        base_dir = os.path.dirname(in_path) or (out_dir or ".")
        out_path = os.path.join(base_dir, f"risk_pre_ssp_ssp_time_c_province_{ssp}_{year}.svg")

    saved = make_fig3_scatter_only(
        df,
        ssp=ssp,
        year=str(year),
        save_path=out_path,
        width_cm=width_cm,
        height_cm=height_cm,
    )
    print("Figure saved:")
    print(saved)
    return saved


if __name__ == "__main__":
    main()
