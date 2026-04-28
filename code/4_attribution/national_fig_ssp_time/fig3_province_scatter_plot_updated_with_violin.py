import os
import re
import glob
from typing import Optional, Tuple, List, Sequence, Dict

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
    "ssp1": "#8694b8",
    "ssp2": "#9570b3",
    "ssp3": "#FB7D5D",
    "ssp4": "#b44f02",
    "ssp5": "#DC3739AA",
    # For averaged plot (fallback)
    "mean": "#4C78A8",
    "avg": "#4C78A8",
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


def prettify_label(s: object) -> str:
    """Make labels nicer for plotting (keeps filenames untouched)."""
    if s is None:
        return ""
    return str(s).replace("_", " ").strip()


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

    year_disp = prettify_label(year)
    ax.set_xlabel("Mean probability (2020)", fontproperties=TNR, color=TEXT_BLACK)
    ax.set_ylabel(f"Mean probability ({year_disp})", fontproperties=TNR, color=TEXT_BLACK)

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    ax.grid(True, axis="both", linewidth=0.5, alpha=0.25)

    # Small in-plot label
    ax.text(
        0.02,
        0.98,
        f"{ssp.upper()} {year_disp}".strip(),
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



def make_fig3_violin_only(
    df: pd.DataFrame,
    ssp: str,
    year: str,
    save_path: str,
    width_cm: float = 6.0,
    height_cm: float = 6.0,
    value_col: str = "delta_mean",
) -> str:
    """Create and save a violin plot for the given scenario (style aligned with the scatter plot).

    By default it visualizes the distribution of `delta_mean` across provinces:
        delta_mean = mean_prob_current - mean_prob_2020
    """
    ssp = (ssp or "ssp5").lower()
    color = SSP_COLOR.get(ssp, "#FB7D5D")

    if value_col not in df.columns:
        raise ValueError(f"Missing column in CSV/DataFrame: {value_col}")

    vals = pd.to_numeric(df[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if vals.size == 0:
        raise ValueError(f"No valid values found for {value_col} to plot.")

    fig_w_in = float(width_cm) / 2.54
    fig_h_in = float(height_cm) / 2.54
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    parts = ax.violinplot(
        dataset=[vals],
        positions=[1],
        widths=0.65,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )

    # Style the violin body
    for body in parts.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor(SPINE_GRAY)
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

    # Style the median/extrema lines
    for k in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if k in parts and parts[k] is not None:
            parts[k].set_color(SPINE_GRAY)
            parts[k].set_linewidth(0.8)

    # Add a subtle jittered point cloud (helps show sample size, keeps style light)
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=0.035, size=vals.size)
    ax.scatter(
        1 + jitter,
        vals,
        s=6,
        color=SPINE_GRAY,
        alpha=0.22,
        linewidth=0,
        zorder=3,
    )

    # Zero reference line
    ax.axhline(0.0, linestyle="--", color=SPINE_GRAY, linewidth=0.8, zorder=1)

    # Y limits with padding (include 0 for context)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    lo = min(vmin, 0.0)
    hi = max(vmax, 0.0)
    pad = max((hi - lo) * 0.08, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)

    # Labels / ticks
    year_disp = prettify_label(year)
    ax.set_xticks([1])
    ax.set_xticklabels([prettify_label(value_col)], fontproperties=TNR, color=TEXT_BLACK)
    ax.set_ylabel("Δ mean probability", fontproperties=TNR, color=TEXT_BLACK)

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)

    # Small in-plot label
    ax.text(
        0.02,
        0.98,
        f"{ssp.upper()} {year_disp}".strip(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontproperties=TNR,
        fontsize=9,
        color=TEXT_BLACK,
    )

    apply_axes_style(ax)

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


def _detect_join_key(dfs: Sequence[pd.DataFrame]) -> Optional[str]:
    """Try to find a common ID column for aligning provinces across scenarios."""
    if not dfs:
        return None

    # Build lower->actual mapping for each df
    maps: List[Dict[str, str]] = []
    for df in dfs:
        maps.append({str(c).lower(): str(c) for c in df.columns})

    candidates = [
        "adcode99",
        "adcode",
        "prov_code",
        "province_code",
        "province",
        "prov",
        "name",
        "province_name",
        "region",
    ]
    for cand in candidates:
        if all(cand in m for m in maps):
            # return the actual column name from the first df
            return maps[0][cand]
    return None


def make_average_df(in_paths: Sequence[str]) -> pd.DataFrame:
    """
    Make an averaged DataFrame over multiple scenario CSVs.

    The average is computed per-province (aligned by a detected join key if possible):
      - mean_prob_2020: mean across inputs
      - mean_prob_current: mean across inputs
      - delta_mean: recomputed as (mean_prob_current - mean_prob_2020)

    If a join key cannot be detected, it falls back to row-order averaging (requires equal lengths).
    """
    if not in_paths:
        raise ValueError("in_paths is empty; cannot compute average.")

    dfs = [load_province_change_csv(p) for p in in_paths]
    join_key = _detect_join_key(dfs)

    # Helper to get a column robustly
    def _col(df: pd.DataFrame, colname: str) -> pd.Series:
        if colname not in df.columns:
            return pd.Series([np.nan] * len(df))
        return pd.to_numeric(df[colname], errors="coerce")

    if join_key and all(join_key in df.columns for df in dfs):
        idx_series = [df[join_key].astype(str) for df in dfs]
        # Use intersection of province IDs to avoid missing values
        common_ids = set(idx_series[0])
        for s in idx_series[1:]:
            common_ids &= set(s)
        common_ids = sorted(common_ids)

        if not common_ids:
            raise ValueError("No common province IDs found across inputs to compute average.")

        m2020 = pd.concat(
            [df.loc[df[join_key].astype(str).isin(common_ids)].set_index(join_key)["mean_prob_2020"] for df in dfs],
            axis=1,
        )
        mcur = pd.concat(
            [df.loc[df[join_key].astype(str).isin(common_ids)].set_index(join_key)["mean_prob_current"] for df in dfs],
            axis=1,
        )

        out = pd.DataFrame(index=m2020.index.astype(str))
        out["mean_prob_2020"] = pd.to_numeric(m2020.mean(axis=1), errors="coerce")
        out["mean_prob_current"] = pd.to_numeric(mcur.mean(axis=1), errors="coerce")
        out["delta_mean"] = out["mean_prob_current"] - out["mean_prob_2020"]

        # Carry n_points from the first file if available (usually identical across scenarios)
        if "n_points" in dfs[0].columns:
            out["n_points"] = (
                dfs[0]
                .loc[dfs[0][join_key].astype(str).isin(common_ids)]
                .set_index(join_key)["n_points"]
                .reindex(out.index)
            )

        # Optional: compute change_rate for potential downstream uses
        with np.errstate(divide="ignore", invalid="ignore"):
            out["change_rate"] = out["delta_mean"] / out["mean_prob_2020"]
            out["change_rate_pct"] = 100.0 * out["change_rate"]

        out = out.reset_index().rename(columns={"index": join_key})
        return out

    # Fallback: align by row order
    lengths = [len(df) for df in dfs]
    if len(set(lengths)) != 1:
        raise ValueError(
            "Cannot detect a common province ID column AND the input files have different row counts. "
            "Please ensure a common ID column exists (e.g., ADCODE99 / province)."
        )

    m2020 = np.vstack([_col(df, "mean_prob_2020").to_numpy(dtype=float) for df in dfs])
    mcur = np.vstack([_col(df, "mean_prob_current").to_numpy(dtype=float) for df in dfs])

    out = pd.DataFrame()
    out["mean_prob_2020"] = np.nanmean(m2020, axis=0)
    out["mean_prob_current"] = np.nanmean(mcur, axis=0)
    out["delta_mean"] = out["mean_prob_current"] - out["mean_prob_2020"]

    if "n_points" in dfs[0].columns:
        out["n_points"] = _col(dfs[0], "n_points").to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["change_rate"] = out["delta_mean"] / out["mean_prob_2020"]
        out["change_rate_pct"] = 100.0 * out["change_rate"]

    return out


def main(
    in_path: Optional[str] = None,
    out_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    ssp: Optional[str] = None,
    year: Optional[str] = None,
    width_cm: float = 6.0,
    height_cm: float = 6.0,
    save_violin: bool = True,
    violin_value_col: str = "delta_mean",
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

    # Also save a violin plot (distribution across provinces)
    if save_violin:
        violin_path = os.path.splitext(out_path)[0] + "_violin.svg"
        saved_v = make_fig3_violin_only(
            df,
            ssp=ssp,
            year=str(year),
            save_path=violin_path,
            width_cm=width_cm,
            height_cm=height_cm,
            value_col=violin_value_col,
        )
        print("Violin figure saved:")
        print(saved_v)

    print("Figure saved:")
    print(saved)
    return saved


def main_batch(
    cases: Sequence[Tuple[str, str, str]],
    out_dir: str,
    width_cm: float = 6.0,
    height_cm: float = 6.0,
    make_average: bool = True,
    average_ssp: str = "mean",
    average_year: str = "mean_5cases",
    average_subdir: Optional[str] = None,
    save_violin: bool = True,
    violin_value_col: str = "delta_mean",
) -> List[str]:
    """
    Batch draw multiple scenarios and (optionally) an averaged plot.

    Args:
        cases: list of (ssp, year, in_path)
        out_dir: the root out_dir (used for resolving paths and average output location)
        make_average: whether to also draw the mean of the provided cases
        average_ssp / average_year: used for the averaged figure label + filename
        average_subdir: if set, averaged figure will be saved to out_dir/average_subdir/
                       otherwise defaults to out_dir/mean/mean_5cases/

    Returns:
        List of saved figure paths (len = len(cases) [+1 if make_average])
    """
    outputs: List[str] = []

    # 1) Draw each case
    for ssp, year, in_path in cases:
        saved = main(
            in_path=in_path,
            out_dir=out_dir,
            ssp=ssp,
            year=year,
            width_cm=width_cm,
            height_cm=height_cm,
            save_violin=save_violin,
            violin_value_col=violin_value_col,
        )
        outputs.append(saved)

    # 2) Draw average
    if make_average:
        in_paths = [p for _, _, p in cases]
        df_avg = make_average_df(in_paths)

        if average_subdir:
            base_dir = os.path.join(out_dir, average_subdir)
        else:
            base_dir = os.path.join(out_dir, average_ssp, average_year)

        os.makedirs(base_dir, exist_ok=True)
        out_path = os.path.join(base_dir, f"risk_pre_ssp_ssp_time_c_province_{average_ssp}_{average_year}.svg")

        saved_avg = make_fig3_scatter_only(
            df_avg,
            ssp=average_ssp,
            year=average_year,
            save_path=out_path,
            width_cm=width_cm,
            height_cm=height_cm,
        )

        if save_violin:
            violin_path = os.path.splitext(out_path)[0] + "_violin.svg"
            saved_v = make_fig3_violin_only(
                df_avg,
                ssp=average_ssp,
                year=average_year,
                save_path=violin_path,
                width_cm=width_cm,
                height_cm=height_cm,
                value_col=violin_value_col,
            )
            print("Average violin figure saved:")
            print(saved_v)

        print("Average figure saved:")
        print(saved_avg)
        outputs.append(saved_avg)

    return outputs


if __name__ == "__main__":
    main()
