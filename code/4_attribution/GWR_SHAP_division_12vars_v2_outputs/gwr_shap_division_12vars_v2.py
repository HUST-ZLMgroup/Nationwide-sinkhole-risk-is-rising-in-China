# -*- coding: utf-8 -*-
"""
Updated GWR-named workflow for 12 variables across China's seven major regions + national.

Core update in v2:
- National figure panel g is NOT a category-colored 12-feature bar chart.
- Instead, panel g stacks each of the 12 features by the seven major regions,
  so each feature is partitioned into regional contributions / shares.
- The small inset in panel g summarizes the three new categories using the same
  seven-region stacking logic.

Figures:
- panels a-f: boxplots of the top-6 ranked features (Disaster=1 vs 0)
- panel g  :
    * subregions: 12-feature category-colored bar chart + category inset
    * national  : 12-feature stacked bars by 7 regions + category stacked inset

Outputs:
- one PNG + one SVG per region
- one feature-importance CSV per region
- national-only extra tables for 7-region feature effects / shares / category means
- notebook clone (.ipynb)
- zipped SVG bundle + zipped full outputs bundle
"""

from __future__ import annotations

import io
import os
import re
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import shap


METHOD_NAMING_NOTE = "GWR naming only; numerical workflow remains RF + TreeSHAP."
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 0) Global style
# =========================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# =========================================================
# 1) Paths & config
# =========================================================
ROOT = Path("/mnt/data/work_shap_v3_gwr_named")
ROOT.mkdir(parents=True, exist_ok=True)

CSV_FILENAME = "AllFeatures_Positive_Negative_balanced_25366_ssp1_cleaned_division.csv"
CSV_PATH = ROOT / CSV_FILENAME
ZIP_PATH = Path("data/raw/input_archive.zip")
OUTPUT_ROOT = ROOT / "division_12vars_v2"
NOTEBOOK_PATH = ROOT / "GWR_SHAP_division_12vars_v2.ipynb"
SCRIPT_PATH = ROOT / "gwr_shap_division_12vars_v2.py"
SVG_ZIP_PATH = ROOT / "division_12vars_v2_svg_only.zip"
FULL_ZIP_PATH = ROOT / "GWR_SHAP_division_12vars_v2_outputs.zip"

TARGET_COL = "Disaster"
DIV_COL = "DIV_EN"
TEST_SIZE = 0.30
RANDOM_STATE = 42
FAST_VALIDATE = False
MAX_ROWS_PER_REGION = None if not FAST_VALIDATE else 2500
MODEL_PARAMS = dict(
    n_estimators=220,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

FEATURES: List[str] = [
    "Distance_to_karst",
    "Depth_to_Bedrock",
    "Distance_to_Fault_m",
    "UrbanFrac_hist_2000_2010_2020",
    "ImperviousIndex_hist_2000_2010_2020",
    "PopTotal_hist_2000_2010_2020",
    "LAI_hist_2000_2010_2020",
    "Precip_hist_2000_2010_2020",
    "Tas_hist_2000_2010_2020",
    "Huss_hist_2000_2010_2020",
    "WTD_hist_2000_2010_2020",
    "HDS_hist_2000_2010_2020",
]

ABBR: Dict[str, str] = {
    "Distance_to_karst": "DK",
    "Depth_to_Bedrock": "DB",
    "Distance_to_Fault_m": "DF",
    "UrbanFrac_hist_2000_2010_2020": "UF",
    "ImperviousIndex_hist_2000_2010_2020": "IP",
    "PopTotal_hist_2000_2010_2020": "PT",
    "LAI_hist_2000_2010_2020": "LAI",
    "Precip_hist_2000_2010_2020": "PR",
    "Tas_hist_2000_2010_2020": "TAS",
    "Huss_hist_2000_2010_2020": "HUSS",
    "WTD_hist_2000_2010_2020": "WTD",
    "HDS_hist_2000_2010_2020": "HDS",
}
ABBR_TO_FULL = {v: k for k, v in ABBR.items()}
FEATURE_ABBRS = [ABBR[f] for f in FEATURES]

CAT_ANTHRO = "Anthropogenic activities"
CAT_CLIMATE = "Climate change"
CAT_HYDRO = "Hydrogeology"

CATEGORY_BY_ABBR: Dict[str, str] = {
    "UF": CAT_ANTHRO,
    "IP": CAT_ANTHRO,
    "PT": CAT_ANTHRO,
    "WTD": CAT_ANTHRO,
    "LAI": CAT_ANTHRO,
    "PR": CAT_CLIMATE,
    "TAS": CAT_CLIMATE,
    "HUSS": CAT_CLIMATE,
    "DK": CAT_HYDRO,
    "DB": CAT_HYDRO,
    "DF": CAT_HYDRO,
    "HDS": CAT_HYDRO,
}
CATEGORY_ORDER = [CAT_ANTHRO, CAT_CLIMATE, CAT_HYDRO]
CATEGORY_SHORT_LABEL = {
    CAT_ANTHRO: "Anthro",
    CAT_CLIMATE: "Climate",
    CAT_HYDRO: "Hydrogeo",
}
CATEGORY_LABEL = {
    CAT_ANTHRO: "Anthropogenic\nactivities",
    CAT_CLIMATE: "Climate\nchange",
    CAT_HYDRO: "Hydrogeology",
}

# Main palette
COLOR_INC = "#C6464A"
COLOR_DEC = "#2F5A73"
COLOR_ANTHRO = "#7A0019"
COLOR_CLIMATE = "#CC1236"
COLOR_HYDRO = "#DCCFAE"
CATEGORY_COLOR = {
    CAT_ANTHRO: COLOR_ANTHRO,
    CAT_CLIMATE: COLOR_CLIMATE,
    CAT_HYDRO: COLOR_HYDRO,
}

# Regional palette for national panel g (close to the provided screenshot)
REGIONS_7 = [
    "Northeast China",
    "North China",
    "East China",
    "Central China",
    "South China",
    "Southwest China",
    "Northwest China",
]
REGION_LABEL = {
    "Northeast China": "Northeast",
    "North China": "North",
    "East China": "East",
    "Central China": "Central",
    "South China": "South",
    "Southwest China": "Southwest",
    "Northwest China": "Northwest",
}
REGION_COLOR = {
    "Northeast China": "#355C7D",
    "North China": "#6C8A9E",
    "East China": "#A7BAC5",
    "Central China": "#C74749",
    "South China": "#D37C63",
    "Southwest China": "#DDB0A0",
    "Northwest China": "#9E9E9E",
}
REGIONS = REGIONS_7 + ["National"]


@dataclass
class RegionResult:
    region_name: str
    region_df: pd.DataFrame
    feature_df: pd.DataFrame
    metrics: Dict[str, float]
    y_test: pd.Series
    region_test: pd.Series
    shap_values: np.ndarray


@dataclass
class NationalStackData:
    effect_df: pd.DataFrame       # rows=7 regions, cols=12 features; values from each regional model mean|SHAP|
    share_df: pd.DataFrame        # same shape; each feature column sums to 1
    cat_mean_df: pd.DataFrame     # rows=7 regions, cols=3 categories; mean feature effect in each category
    feature_total: pd.Series      # summed effect over the 7 regions
    feature_order: List[str]


# =========================================================
# 2) Utilities
# =========================================================
def safe_name(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s


def compact_num(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    av = abs(v)
    if av >= 1e5 or (0 < av < 1e-2):
        return f"{v:.1e}"
    if av >= 1e3:
        return f"{v:,.0f}"
    if av >= 100:
        return f"{v:,.1f}"
    if av >= 10:
        return f"{v:,.2f}"
    return f"{v:.3f}"


def apply_sci_y(ax: plt.Axes) -> None:
    vals: List[float] = []
    for artist in ax.lines:
        yd = artist.get_ydata(orig=False)
        if yd is not None and len(yd):
            vals.extend(np.asarray(yd, dtype=float).ravel().tolist())
    if not vals:
        return
    vmax = np.nanmax(np.abs(vals))
    if (vmax >= 1e4) or (0 < vmax <= 1e-2):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))


def style_axes_box(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", length=3, width=1)


def style_axes_bar(ax: plt.Axes) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1)
    ax.tick_params(axis="both", which="major", length=3, width=1)


def add_panel_letter(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.04, fontsize: int = 13) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontweight="bold", fontsize=fontsize)


def ensure_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def maybe_sample_region(df_region: pd.DataFrame, region_name: str) -> pd.DataFrame:
    if MAX_ROWS_PER_REGION is None or len(df_region) <= MAX_ROWS_PER_REGION:
        return df_region.copy()
    sampled = (
        df_region.groupby(TARGET_COL, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), MAX_ROWS_PER_REGION // 2), random_state=RANDOM_STATE))
        .sort_index()
        .copy()
    )
    print(f"[FAST_VALIDATE] {region_name}: sampled {len(sampled):,} / {len(df_region):,} rows")
    return sampled


def boxplot_two_groups(ax: plt.Axes, data_inc: np.ndarray, data_dec: np.ndarray, ylabel: str) -> None:
    data_inc = np.asarray(pd.to_numeric(pd.Series(data_inc), errors="coerce"), dtype=float)
    data_dec = np.asarray(pd.to_numeric(pd.Series(data_dec), errors="coerce"), dtype=float)
    data_inc = data_inc[np.isfinite(data_inc)]
    data_dec = data_dec[np.isfinite(data_dec)]

    if len(data_inc) == 0 or len(data_dec) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_ylabel(ylabel)
        style_axes_box(ax)
        return

    b = ax.boxplot(
        [data_inc, data_dec],
        tick_labels=["Increased", "Decreased"],
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        widths=0.70,
        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="black", markersize=5),
        medianprops=dict(color="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )
    for patch, c in zip(b["boxes"], [COLOR_INC, COLOR_DEC]):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    ax.set_ylabel(ylabel)
    style_axes_box(ax)
    apply_sci_y(ax)

    m1 = np.nanmean(data_inc)
    m2 = np.nanmean(data_dec)
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if np.isfinite(y1 - y0) and (y1 > y0) else max(np.nanmax(data_inc), np.nanmax(data_dec), 1.0)
    ax.set_ylim(y0, y1 + 0.16 * yr)
    ax.text(1, y1 + 0.035 * yr, compact_num(m1), ha="center", va="bottom", fontsize=8)
    ax.text(2, y1 + 0.035 * yr, compact_num(m2), ha="center", va="bottom", fontsize=8)


def load_dataframe() -> pd.DataFrame:
    direct_candidates = [
        CSV_PATH,
        Path("/mnt/data") / CSV_FILENAME,
        Path.cwd() / CSV_FILENAME,
    ]
    for cand in direct_candidates:
        if cand.exists():
            print(f"Loading CSV directly: {cand}")
            return pd.read_csv(cand)

    if ZIP_PATH.exists():
        print(f"Loading CSV from ZIP: {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            if CSV_FILENAME not in zf.namelist():
                raise FileNotFoundError(f"{CSV_FILENAME} not found inside {ZIP_PATH}")
            with zf.open(CSV_FILENAME) as f:
                return pd.read_csv(f)

    raise FileNotFoundError(
        f"Could not locate {CSV_FILENAME}. Checked direct paths and ZIP {ZIP_PATH}."
    )


# =========================================================
# 3) Model / SHAP
# =========================================================
def train_one_region(df_all: pd.DataFrame, region_name: str) -> RegionResult:
    if region_name == "National":
        df_region = df_all.loc[df_all[DIV_COL].isin(REGIONS_7)].copy()
    else:
        df_region = df_all.loc[df_all[DIV_COL] == region_name].copy()

    df_region = maybe_sample_region(df_region, region_name)
    use_cols = [DIV_COL, TARGET_COL] + FEATURES
    missing = [c for c in use_cols if c not in df_region.columns]
    if missing:
        raise KeyError(f"{region_name} missing columns: {missing}")

    df_region = ensure_numeric_frame(df_region, [TARGET_COL] + FEATURES)
    df_region = df_region[use_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df_region[TARGET_COL] = df_region[TARGET_COL].astype(int)

    if df_region[TARGET_COL].nunique() < 2:
        raise ValueError(f"{region_name}: target has only one class after cleaning.")

    X = df_region[FEATURES].copy()
    y = df_region[TARGET_COL].copy()
    div = df_region[DIV_COL].copy()

    X_train, X_test, y_train, y_test, div_train, div_test = train_test_split(
        X,
        y,
        div,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "AUC": float(roc_auc_score(y_test, prob)),
        "ACC": float(accuracy_score(y_test, pred)),
        "F1": float(f1_score(y_test, pred)),
        "n_total": int(len(df_region)),
        "n_test": int(len(X_test)),
    }

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_test, check_additivity=False, approximate=True)
    shap_raw = np.asarray(shap_raw)
    if shap_raw.ndim == 3 and shap_raw.shape[-1] == 2:
        shap_pos = shap_raw[..., 1]
    elif shap_raw.ndim == 3 and shap_raw.shape[0] == 2:
        shap_pos = shap_raw[1]
    else:
        shap_pos = np.asarray(shap_raw)

    mean_abs = np.abs(shap_pos).mean(axis=0)
    signs: List[str] = []
    for j in range(len(FEATURES)):
        xj = X_test.iloc[:, j].to_numpy(dtype=float)
        sj = shap_pos[:, j]
        if np.all(np.isfinite(xj)) and np.all(np.isfinite(sj)) and np.nanstd(xj) > 0 and np.nanstd(sj) > 0:
            r = np.corrcoef(xj, sj)[0, 1]
        else:
            r = np.nan
        signs.append("+" if np.isfinite(r) and r >= 0 else "-")

    imp = pd.DataFrame({
        "Feature_full": FEATURES,
        "Feature": [ABBR[f] for f in FEATURES],
        "Mean_SHAP": mean_abs,
        "Sign": signs,
    })
    imp["Category"] = imp["Feature"].map(CATEGORY_BY_ABBR)
    imp["Category_label"] = imp["Category"].map(CATEGORY_LABEL)
    imp = imp.sort_values("Mean_SHAP", ascending=False, kind="mergesort").reset_index(drop=True)

    return RegionResult(
        region_name=region_name,
        region_df=df_region,
        feature_df=imp,
        metrics=metrics,
        y_test=y_test.reset_index(drop=True),
        region_test=div_test.reset_index(drop=True),
        shap_values=np.asarray(shap_pos),
    )


# =========================================================
# 4) National panel-g synthesis from the 7 regional models
# =========================================================
def build_national_stack_data(results_map: Dict[str, RegionResult]) -> NationalStackData:
    effect_df = pd.DataFrame(index=REGIONS_7, columns=FEATURE_ABBRS, dtype=float)
    for region in REGIONS_7:
        imp = results_map[region].feature_df.set_index("Feature")["Mean_SHAP"]
        effect_df.loc[region] = imp.reindex(FEATURE_ABBRS).astype(float)
    effect_df = effect_df.fillna(0.0)

    feature_total = effect_df.sum(axis=0).sort_values(ascending=False, kind="mergesort")
    feature_order = feature_total.index.tolist()
    share_df = effect_df.div(effect_df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    share_df = share_df[feature_order]
    effect_df = effect_df[feature_order]

    cat_mean_df = pd.DataFrame(index=REGIONS_7, columns=CATEGORY_ORDER, dtype=float)
    for cat in CATEGORY_ORDER:
        feats = [f for f in FEATURE_ABBRS if CATEGORY_BY_ABBR[f] == cat]
        cat_mean_df[cat] = effect_df[feats].mean(axis=1)
    cat_mean_df = cat_mean_df[CATEGORY_ORDER]

    return NationalStackData(
        effect_df=effect_df,
        share_df=share_df,
        cat_mean_df=cat_mean_df,
        feature_total=feature_total,
        feature_order=feature_order,
    )


# =========================================================
# 5) Plot helpers
# =========================================================
def category_summary_from_imp(imp: pd.DataFrame) -> pd.Series:
    out = (
        imp.groupby("Category", sort=False)["Mean_SHAP"]
        .mean()
        .reindex(CATEGORY_ORDER)
        .fillna(0.0)
    )
    return out


def save_png_svg(fig: plt.Figure, out_folder: Path, stem: str, dpi: int = 300) -> Tuple[Path, Path]:
    out_folder.mkdir(parents=True, exist_ok=True)
    png_path = out_folder / f"{stem}.png"
    svg_path = out_folder / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def draw_category_inset(axg: plt.Axes, imp: pd.DataFrame, bounds: Sequence[float]) -> None:
    axh = axg.inset_axes(bounds)
    summary = category_summary_from_imp(imp)
    x = np.arange(len(summary))
    bars = axh.bar(x, summary.values, width=0.55)
    for rect, cat in zip(bars, summary.index):
        rect.set_facecolor(CATEGORY_COLOR[cat])
        rect.set_edgecolor("black")
        rect.set_linewidth(0.6)
    axh.set_xticks(x)
    axh.set_xticklabels([CATEGORY_LABEL[c] for c in summary.index], rotation=90, ha="center")
    axh.tick_params(axis="both", which="major", length=2, width=0.8)
    axh.yaxis.set_major_locator(MaxNLocator(nbins=3))
    for sp in axh.spines.values():
        sp.set_linewidth(1)
    axh.set_facecolor("white")


def draw_national_main_stacks(
    axg: plt.Axes,
    national_stack: NationalStackData,
    sign_map: Dict[str, str],
) -> None:
    x = np.arange(len(national_stack.feature_order))
    bottom = np.zeros(len(national_stack.feature_order), dtype=float)

    for region in REGIONS_7:
        vals = national_stack.effect_df.loc[region, national_stack.feature_order].to_numpy(dtype=float)
        axg.bar(
            x,
            vals,
            bottom=bottom,
            width=0.62,
            color=REGION_COLOR[region],
            edgecolor="none",
            label=REGION_LABEL[region],
        )
        bottom += vals

    axg.bar(
        x,
        national_stack.feature_total.values,
        width=0.62,
        facecolor="none",
        edgecolor="black",
        linewidth=0.7,
    )

    axg.set_ylabel("|Total effect|")
    axg.set_xticks(x)
    axg.set_xticklabels(national_stack.feature_order, rotation=90, ha="center")
    style_axes_bar(axg)

    ymax = float(national_stack.feature_total.max()) if len(national_stack.feature_total) else 1.0
    axg.set_ylim(0, ymax * 1.34)

    for i, feat in enumerate(national_stack.feature_order):
        axg.text(
            i,
            float(national_stack.feature_total.loc[feat]) + 0.03 * ymax,
            sign_map.get(feat, ""),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axg.legend(
        frameon=False,
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.01),
        handlelength=1.8,
        columnspacing=0.8,
    )


def draw_national_category_inset(axg: plt.Axes, national_stack: NationalStackData, bounds: Sequence[float]) -> None:
    axh = axg.inset_axes(bounds)
    x = np.arange(len(CATEGORY_ORDER))
    bottom = np.zeros(len(CATEGORY_ORDER), dtype=float)

    for region in REGIONS_7:
        vals = national_stack.cat_mean_df.loc[region, CATEGORY_ORDER].to_numpy(dtype=float)
        axh.bar(
            x,
            vals,
            bottom=bottom,
            width=0.55,
            color=REGION_COLOR[region],
            edgecolor="none",
        )
        bottom += vals

    axh.bar(
        x,
        national_stack.cat_mean_df[CATEGORY_ORDER].sum(axis=0).to_numpy(dtype=float),
        width=0.55,
        facecolor="none",
        edgecolor="black",
        linewidth=0.6,
    )
    axh.set_xticks(x)
    axh.set_xticklabels([CATEGORY_SHORT_LABEL[c] for c in CATEGORY_ORDER], rotation=90, ha="center")
    axh.tick_params(axis="both", which="major", length=2, width=0.8)
    axh.yaxis.set_major_locator(MaxNLocator(nbins=3))
    for sp in axh.spines.values():
        sp.set_linewidth(1)
    axh.set_facecolor("white")


# =========================================================
# 6) Figure generation
# =========================================================
def plot_region_figure(
    result: RegionResult,
    results_map: Dict[str, RegionResult],
    out_root: Path,
    national_stack: NationalStackData | None = None,
) -> Tuple[Path, Path]:
    imp = result.feature_df.copy()
    region_name = result.region_name

    region_tag = safe_name(region_name)
    out_folder = out_root / region_tag
    out_folder.mkdir(parents=True, exist_ok=True)

    # save main importance table from the region/national model itself
    imp_csv = out_folder / f"{region_tag}_importance_table.csv"
    imp.to_csv(imp_csv, index=False, encoding="utf-8-sig")

    # For national: save the 7-region synthesis tables used in panel g
    if region_name == "National" and national_stack is not None:
        national_stack.effect_df.to_csv(
            out_folder / f"{region_tag}_7region_feature_effect_table.csv",
            encoding="utf-8-sig",
        )
        national_stack.share_df.to_csv(
            out_folder / f"{region_tag}_7region_feature_share_table.csv",
            encoding="utf-8-sig",
        )
        national_stack.cat_mean_df.to_csv(
            out_folder / f"{region_tag}_7region_category_mean_effect_table.csv",
            encoding="utf-8-sig",
        )
        summary_df = pd.DataFrame({
            "Feature": national_stack.feature_order,
            "Total_effect": national_stack.feature_total.reindex(national_stack.feature_order).values,
            "Sign_from_national_model": [
                result.feature_df.set_index("Feature").loc[f, "Sign"] if f in result.feature_df["Feature"].values else ""
                for f in national_stack.feature_order
            ],
            "Category": [CATEGORY_BY_ABBR[f] for f in national_stack.feature_order],
        })
        summary_df.to_csv(
            out_folder / f"{region_tag}_7region_feature_total_effect_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # National: top6 boxplots follow the 7-region stacked ranking from panel g
    if region_name == "National" and national_stack is not None:
        top6_abbr = national_stack.feature_order[:6]
        top6 = pd.DataFrame({
            "Feature": top6_abbr,
            "Feature_full": [ABBR_TO_FULL[a] for a in top6_abbr],
        })
    else:
        top6 = imp.head(6).copy()[["Feature", "Feature_full"]]

    fig = plt.figure(figsize=(8.4, 7.5))
    gs = GridSpec(
        nrows=3,
        ncols=3,
        height_ratios=[1.0, 1.0, 1.50],
        hspace=0.48,
        wspace=0.55,
        figure=fig,
    )

    letters = list("abcdef")
    for i in range(6):
        r = 0 if i < 3 else 1
        c = i % 3
        ax = fig.add_subplot(gs[r, c])
        feat_full = top6.iloc[i]["Feature_full"]
        feat_abbr = top6.iloc[i]["Feature"]

        inc = result.region_df.loc[result.region_df[TARGET_COL] == 1, feat_full].to_numpy(dtype=float)
        dec = result.region_df.loc[result.region_df[TARGET_COL] == 0, feat_full].to_numpy(dtype=float)
        boxplot_two_groups(ax, inc, dec, ylabel=feat_abbr)
        add_panel_letter(ax, letters[i], x=-0.18, y=1.04, fontsize=13)

    axg = fig.add_subplot(gs[2, :])
    add_panel_letter(axg, "g", x=-0.10, y=1.02, fontsize=13)

    if region_name == "National" and national_stack is not None:
        sign_map = result.feature_df.set_index("Feature")["Sign"].to_dict()
        draw_national_main_stacks(axg, national_stack, sign_map=sign_map)
        draw_national_category_inset(axg, national_stack, bounds=[0.64, 0.58, 0.32, 0.36])
    else:
        plot_df = imp.copy()
        x = np.arange(len(plot_df))
        colors = [CATEGORY_COLOR[cat] for cat in plot_df["Category"]]
        bars = axg.bar(x, plot_df["Mean_SHAP"].values, width=0.58)
        for rect, col in zip(bars, colors):
            rect.set_facecolor(col)
            rect.set_edgecolor("black")
            rect.set_linewidth(0.6)

        ymax = float(plot_df["Mean_SHAP"].max()) if len(plot_df) else 1.0
        axg.set_ylim(0, ymax * 1.85)
        axg.set_ylabel("Mean |SHAP| value")
        axg.set_xticks(x)
        axg.set_xticklabels(plot_df["Feature"].tolist(), rotation=90, ha="center")
        style_axes_bar(axg)

        for i, (v, sgn) in enumerate(zip(plot_df["Mean_SHAP"].values, plot_df["Sign"].values)):
            axg.text(i, v + 0.028 * ymax, sgn, ha="center", va="bottom", fontsize=10)

        stats_text = (
            f"AUC={result.metrics['AUC']:.2f}\n"
            f"ACC={result.metrics['ACC']:.2f}\n"
            f"F1={result.metrics['F1']:.2f}\n"
            f"n={result.metrics['n_total']:,}"
        )
        axg.text(0.03, 0.95, stats_text, transform=axg.transAxes, ha="left", va="top")

        legend_handles = [Patch(facecolor=CATEGORY_COLOR[c], edgecolor="black", label=c) for c in CATEGORY_ORDER]
        axg.legend(
            handles=legend_handles,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.01),
            handlelength=1.2,
            columnspacing=1.0,
        )
        draw_category_inset(axg, imp, bounds=[0.73, 0.55, 0.22, 0.32])

    fig.suptitle(region_name, y=0.995, fontweight="bold")
    stem = f"{region_tag}_Fig_Box_SHAP_7panels_12vars_v2"
    return save_png_svg(fig, out_folder, stem)


# =========================================================
# 7) Packaging / notebook
# =========================================================
def zip_matching_files(root: Path, pattern_suffixes: Tuple[str, ...], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(root.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in pattern_suffixes:
                zf.write(fp, fp.relative_to(root.parent))


def zip_full_outputs(root: Path, extra_files: Sequence[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(root.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(root.parent))
        for fp in extra_files:
            if fp.exists():
                zf.write(fp, fp.name)


def build_notebook(script_text: str) -> None:
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# GWR-named division workflow (12 variables, corrected national panel g)\n\n"
            "This package aligns the file naming with the broader GWR-based proxy-model workflow.\n\n"
            "Important note:\n\n"
            "- The plotting and numerical results in this notebook are unchanged from the previous version.\n"
            "- The underlying modelling block below is still the existing RF + TreeSHAP implementation.\n"
            "- A true GWR / GWLR replacement would require rewriting the model-fitting and interpretation sections rather than only renaming files.\n\n"
            "The national panel g remains the corrected 12 features x 7 regions stacked-share mapping."
        ),
        nbf.v4.new_code_cell(script_text),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.13"}
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =========================================================
# 8) Main
# =========================================================
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    required_cols = [TARGET_COL, DIV_COL] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    print(f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print(f"Output root: {OUTPUT_ROOT}")

    results_map: Dict[str, RegionResult] = {}
    for region in REGIONS:
        print(f"\n=== Processing: {region} ===")
        res = train_one_region(df, region)
        results_map[region] = res
        print(
            f"AUC={res.metrics['AUC']:.3f}, ACC={res.metrics['ACC']:.3f}, "
            f"F1={res.metrics['F1']:.3f}, n={res.metrics['n_total']:,}"
        )

    national_stack = build_national_stack_data(results_map)

    for region in REGIONS:
        png_path, svg_path = plot_region_figure(
            results_map[region],
            results_map=results_map,
            out_root=OUTPUT_ROOT,
            national_stack=national_stack if region == "National" else None,
        )
        print(f"Saved figure: {svg_path}")

    build_notebook(SCRIPT_PATH.read_text(encoding="utf-8"))
    zip_matching_files(OUTPUT_ROOT, (".svg",), SVG_ZIP_PATH)
    zip_full_outputs(OUTPUT_ROOT, [NOTEBOOK_PATH, SCRIPT_PATH], FULL_ZIP_PATH)

    print("\nDone.")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"SVG zip : {SVG_ZIP_PATH}")
    print(f"Full zip: {FULL_ZIP_PATH}")


if __name__ == "__main__":
    main()
