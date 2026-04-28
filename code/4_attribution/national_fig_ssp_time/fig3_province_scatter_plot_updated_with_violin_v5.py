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

SPINE_GRAY = "#4D4D4D"
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
    """Load the province change CSV (headered) and coerce known numeric columns."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Trim header whitespace (common when CSVs are assembled in Excel)
    df.columns = [str(c).strip() for c in df.columns]

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
    if d.empty:
        counts = {c: int(df[c].notna().sum()) if c in df.columns else 0 for c in required}
        raise ValueError(
            "No valid rows after dropping NaNs for required columns "
            f"{required}. Non-NaN counts: {counts}"
        )
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
    """Create and save a single-factor violin plot (one group) for the given scenario.

    Style is inspired by the reference: white background, dashed horizontal guide lines,
    unfilled violin outline + boxplot overlay + jittered points + mean annotation.
    """
    from matplotlib.ticker import AutoMinorLocator

    ssp = (ssp or "ssp5").lower()
    point_color = SSP_COLOR.get(ssp, "#FB7D5D")

    # Local palette (close to the reference)
    colors = {
        "BG_WHITE": "white",
        "GREY50": "#999999",
        "BLACK": "#282724",
        "GREY_DARK": "#747473",
        "RED_DARK": "#850e00",
    }

    if value_col not in df.columns:
        raise ValueError(f"Missing column in CSV/DataFrame: {value_col}")

    vals = (
        pd.to_numeric(df[value_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if vals.size == 0:
        raise ValueError(f"No valid values found for {value_col} to plot.")

    # --- Figure canvas (keeps 6cm x 6cm exactly) ---
    fig_w_in = float(width_cm) / 2.54
    fig_h_in = float(height_cm) / 2.54
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor(colors["BG_WHITE"])
    ax.set_facecolor(colors["BG_WHITE"])

    # --- Limits (include 0 for context) ---
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    lo = min(vmin, 0.0)
    hi = max(vmax, 0.0)
    pad = max((hi - lo) * 0.10, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)

    # --- Y ticks + dashed horizontal guide lines (drawn behind) ---
    ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.canvas.draw_idle()
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        ax.axhline(
            y,
            color=colors["GREY50"],
            ls=(0, (5, 5)),
            alpha=0.8,
            linewidth=0.8,
            zorder=0,
        )

    # --- Violin outline (no fill) ---
    violins = ax.violinplot(
        [vals],
        positions=[0],
        widths=0.45,
        bw_method="silverman",
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in violins.get("bodies", []):
        pc.set_facecolor("none")
        pc.set_edgecolor(colors["BLACK"])
        pc.set_linewidth(1.5)

    # --- Boxplot overlay (median / IQR / whiskers) ---
    ax.boxplot(
        [vals],
        positions=[0],
        widths=0.18,
        showfliers=False,
        showcaps=False,
        medianprops=dict(linewidth=2.4, color=colors["GREY_DARK"]),
        whiskerprops=dict(linewidth=1.5, color=colors["GREY_DARK"]),
        boxprops=dict(linewidth=1.5, color=colors["GREY_DARK"]),
    )

    # --- Jittered points (t-distribution jitter like the reference) ---
    rng = np.random.default_rng(42)
    jitter = 0.04
    x_j = 0.0 + rng.standard_t(df=6, size=vals.size) * jitter
    ax.scatter(
        x_j,
        vals,
        s=18,
        color=point_color,
        alpha=0.35,
        linewidth=0,
        zorder=2,
    )

    # --- 0 reference line ---
    ax.axhline(0.0, color=colors["BLACK"], linewidth=0.9, ls=":", alpha=0.9, zorder=1)

    # --- Mean annotation (red dot + short dash-dot + label) ---
    mean = float(np.nanmean(vals))
    ax.scatter(0.0, mean, s=45, color=colors["RED_DARK"], zorder=4)
    ax.plot([0.0, 0.25], [mean, mean], ls="dashdot", color=colors["BLACK"], linewidth=1.0, zorder=4)
    ax.text(
        0.25,
        mean,
        f"$\\hat{{\\mu}}$ = {mean:.3f}",
        fontsize=7.5,
        va="center",
        ha="left",
        color=colors["BLACK"],
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.15),
        zorder=10,
    )

    # --- Axes cosmetics (match the reference feel) ---
    # Show full border (top/right included) with uniform thickness = 1
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_color(SPINE_GRAY)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(axis="both", direction="out", length=4, width=1.0, color=SPINE_GRAY)
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.tick_params(which="minor", length=2.5, width=0.6, color=SPINE_GRAY)

    year_disp = prettify_label(year)
    xt = f"{ssp.upper()} {year_disp}\n(n={vals.size})".strip()
    ax.set_xticks([0])
    ax.set_xticklabels([xt], fontproperties=TNR, color=TEXT_BLACK, fontsize=8)

    if value_col == "delta_mean":
        ax.set_ylabel("Δ mean probability", fontproperties=TNR, color=TEXT_BLACK)
    else:
        ax.set_ylabel(prettify_label(value_col), fontproperties=TNR, color=TEXT_BLACK)

    # Give the annotation room on the right
    # Center the single violin in the horizontal axis (leave balanced space)
    ax.set_xlim(-0.85, 0.85)

    # Keep the canvas size exactly 6cm x 6cm; control margins manually
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.98)

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

    Notes
    -----
    This function is careful about index alignment. In earlier versions, converting the index
    to string on only one side could produce all-NaN results due to pandas' label alignment.
    """
    if not in_paths:
        raise ValueError("in_paths is empty; cannot compute average.")

    dfs = [load_province_change_csv(p) for p in in_paths]
    join_key = _detect_join_key(dfs)

    def _norm_id_series(s: pd.Series) -> pd.Series:
        # Convert to clean string IDs; avoid keeping NaN as literal 'nan'
        out = s.astype(str).str.strip()
        out = out.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "<NA>": np.nan})
        return out

    def _col_numeric(df: pd.DataFrame, colname: str) -> pd.Series:
        if colname not in df.columns:
            return pd.Series([np.nan] * len(df))
        return pd.to_numeric(df[colname], errors="coerce")

    # -------- per-province aligned average (preferred) --------
    if join_key and all(join_key in df.columns for df in dfs):
        id_series_list = [_norm_id_series(df[join_key]).dropna() for df in dfs]

        common_ids = set(id_series_list[0].tolist())
        for s in id_series_list[1:]:
            common_ids &= set(s.tolist())
        common_ids = sorted(common_ids)

        if not common_ids:
            raise ValueError(
                "No common province IDs found across inputs to compute average. "
                f"Detected join key: {join_key}"
            )

        m2020_cols = []
        mcur_cols = []
        npts_ref = None

        for df in dfs:
            tmp = df.copy()
            tmp["_join_id"] = _norm_id_series(tmp[join_key])
            tmp = tmp.dropna(subset=["_join_id"])

            # Handle potential duplicates by averaging within the same province ID
            g = tmp.groupby("_join_id", sort=False)

            s2020 = pd.to_numeric(g["mean_prob_2020"].mean(), errors="coerce").reindex(common_ids)
            scur = pd.to_numeric(g["mean_prob_current"].mean(), errors="coerce").reindex(common_ids)

            m2020_cols.append(s2020)
            mcur_cols.append(scur)

            if npts_ref is None and "n_points" in tmp.columns:
                npts_ref = pd.to_numeric(g["n_points"].mean(), errors="coerce").reindex(common_ids)

        m2020 = pd.concat(m2020_cols, axis=1)
        mcur = pd.concat(mcur_cols, axis=1)

        out = pd.DataFrame(index=pd.Index(common_ids, name=join_key))
        out["mean_prob_2020"] = m2020.mean(axis=1, skipna=True)
        out["mean_prob_current"] = mcur.mean(axis=1, skipna=True)
        out["delta_mean"] = out["mean_prob_current"] - out["mean_prob_2020"]

        if npts_ref is not None:
            out["n_points"] = npts_ref

        with np.errstate(divide="ignore", invalid="ignore"):
            out["change_rate"] = out["delta_mean"] / out["mean_prob_2020"]
            out["change_rate_pct"] = 100.0 * out["change_rate"]

        return out.reset_index()

    # -------- fallback: align by row order --------
    lengths = [len(df) for df in dfs]
    if len(set(lengths)) != 1:
        raise ValueError(
            "Cannot detect a common province ID column AND the input files have different row counts. "
            "Please ensure a common ID column exists (e.g., ADCODE99 / province)."
        )

    m2020 = np.vstack([_col_numeric(df, "mean_prob_2020").to_numpy(dtype=float) for df in dfs])
    mcur = np.vstack([_col_numeric(df, "mean_prob_current").to_numpy(dtype=float) for df in dfs])

    out = pd.DataFrame()
    out["mean_prob_2020"] = np.nanmean(m2020, axis=0)
    out["mean_prob_current"] = np.nanmean(mcur, axis=0)
    out["delta_mean"] = out["mean_prob_current"] - out["mean_prob_2020"]

    if "n_points" in dfs[0].columns:
        out["n_points"] = _col_numeric(dfs[0], "n_points").to_numpy(dtype=float)

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
