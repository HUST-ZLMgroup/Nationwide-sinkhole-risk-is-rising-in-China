from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


THRESHOLD = 0.35
MAX_MITIGATION_RR_BAND = 0.90
COST_EFFICIENCY_BAND = 0.90

MAP_EN = {
    "达标优先": "Target-first",
    "极限减灾": "Maximum mitigation",
    "性价比优先": "Cost-efficiency-first",
    "平衡折中": "Balanced compromise",
}
ORDER_CN = ["达标优先", "极限减灾", "性价比优先", "平衡折中"]
ORDER = [MAP_EN[x] for x in ORDER_CN]

COLORS = {
    "Target-first": "#C67C6B",
    "Maximum mitigation": "#8E6558",
    "Cost-efficiency-first": "#D7A38C",
    "Balanced compromise": "#B7A08D",
}
GREY = "#6E6A67"
LIGHT_GREY = "#CFC7C1"
EPS = 1e-9

# Match the knee-based post-Pareto selector used in
# `1_nsga2_optimize_gwr_resume_rrfix_10km_v4.ipynb` for bal11.
KNEE_RR_WEIGHT = 1.00
KNEE_COST_WEIGHT = 0.50
KNEE_DROP_WEIGHT = 6.00
KNEE_BALANCE_LOG_WEIGHT = 2.00
KNEE_BALANCE_SHARE_WEIGHT = 4.00
KNEE_ISA_SHARE_WEIGHT = 0.00
KNEE_LAI_SHARE_EXCESS_WEIGHT = 0.00
KNEE_CAP_EXCESS_WEIGHT = 100.0
KNEE_MIN_RR_FRACTION = 0.00

ISA_NET_COST_CAP_USD = 2.0e8
LAI_NET_COST_CAP_USD = 2.5e7
WTD_NET_COST_CAP_USD = 5.0e7

FS_TICK = 14
FS_SMALL = 12
FS_ANNOT = 13
FS_AXIS = 17
FS_PANEL_TITLE = 18
FS_PANEL_LABEL = 21
FS_FIG_TITLE = 20
FS_LEGEND = 11
FS_SUP_TITLE = 14


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
        "text.color": "#2F2F2F",
    }
)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    v = values[mask].to_numpy(dtype=float)
    w = weights[mask].to_numpy(dtype=float)
    if np.allclose(w.sum(), 0.0):
        return float(np.mean(v))
    return float(np.average(v, weights=w))


def weighted_share(flag: pd.Series, weights: pd.Series) -> float:
    flag = flag.astype(float)
    return weighted_mean(flag, weights)


def component_cap_excess_ratio(cost_parts: np.ndarray) -> np.ndarray:
    cost_parts = np.maximum(np.asarray(cost_parts, dtype=float), 0.0)
    if cost_parts.ndim == 1:
        cost_parts = cost_parts.reshape(1, -1)
    caps = np.array(
        [
            max(float(ISA_NET_COST_CAP_USD), EPS),
            max(float(WTD_NET_COST_CAP_USD), EPS),
            max(float(LAI_NET_COST_CAP_USD), EPS),
        ],
        dtype=float,
    ).reshape(1, 3)
    excess = np.maximum(cost_parts - caps, 0.0) / caps
    return np.sum(excess, axis=1)


def component_log_balance_penalty(cost_parts: np.ndarray) -> np.ndarray:
    cost_parts = np.maximum(np.asarray(cost_parts, dtype=float), 0.0)
    if cost_parts.ndim == 1:
        cost_parts = cost_parts.reshape(1, -1)
    log_costs = np.log1p(cost_parts)
    return np.max(log_costs, axis=1) - np.min(log_costs, axis=1)


def component_share_balance_penalty(cost_parts: np.ndarray) -> np.ndarray:
    cost_parts = np.maximum(np.asarray(cost_parts, dtype=float), 0.0)
    if cost_parts.ndim == 1:
        cost_parts = cost_parts.reshape(1, -1)
    total = np.sum(cost_parts, axis=1, keepdims=True)
    shares = np.divide(cost_parts, total, out=np.zeros_like(cost_parts), where=total > EPS)
    return np.mean((shares - (1.0 / 3.0)) ** 2, axis=1)


def pick_knee_solution_index(sub: pd.DataFrame) -> int:
    rr = sub["RR"].to_numpy(dtype=float)
    cost = sub["Cost_total"].to_numpy(dtype=float)
    drop = (
        sub["risk_level_drop"].to_numpy(dtype=float)
        if "risk_level_drop" in sub.columns
        else np.zeros_like(rr)
    )
    cost_parts = sub[["Cost_ISA", "Cost_WT", "Cost_LAI"]].to_numpy(dtype=float)
    idx_all = np.arange(rr.shape[0])

    rr_max_all = float(np.nanmax(rr)) if rr.size else 0.0
    if rr_max_all > EPS and float(KNEE_MIN_RR_FRACTION) > 0.0:
        candidate_mask = rr >= (float(KNEE_MIN_RR_FRACTION) * rr_max_all)
        if not np.any(candidate_mask):
            candidate_mask = np.ones_like(rr, dtype=bool)
    else:
        candidate_mask = np.ones_like(rr, dtype=bool)

    rr_c = rr[candidate_mask]
    cost_c = cost[candidate_mask]
    drop_c = drop[candidate_mask]
    idx_c = idx_all[candidate_mask]

    rr_min, rr_max = float(np.min(rr_c)), float(np.max(rr_c))
    c_min, c_max = float(np.min(cost_c)), float(np.max(cost_c))
    rr_norm = (rr_c - rr_min) / (rr_max - rr_min + EPS)
    c_norm = (cost_c - c_min) / (c_max - c_min + EPS)

    drop_max = float(np.max(drop_c)) if drop_c.size else 0.0
    drop_norm = drop_c / max(drop_max, 1.0)
    score = (
        float(KNEE_RR_WEIGHT) * (1.0 - rr_norm) ** 2
        + float(KNEE_COST_WEIGHT) * (c_norm) ** 2
        - float(KNEE_DROP_WEIGHT) * drop_norm
    )

    cost_parts_c = cost_parts[candidate_mask]
    log_bal = component_log_balance_penalty(cost_parts_c)
    share_bal = component_share_balance_penalty(cost_parts_c)
    cap_excess = component_cap_excess_ratio(cost_parts_c)
    total_parts = np.sum(np.maximum(cost_parts_c, 0.0), axis=1, keepdims=True)
    cost_shares = np.divide(
        np.maximum(cost_parts_c, 0.0),
        total_parts,
        out=np.zeros_like(cost_parts_c, dtype=float),
        where=total_parts > EPS,
    )
    isa_share = cost_shares[:, 0]
    lai_share = cost_shares[:, 2]

    lb_min, lb_max = float(np.min(log_bal)), float(np.max(log_bal))
    if lb_max > lb_min:
        lb_norm = (log_bal - lb_min) / (lb_max - lb_min + EPS)
        score = score + float(KNEE_BALANCE_LOG_WEIGHT) * (lb_norm**2)

    sb_min, sb_max = float(np.min(share_bal)), float(np.max(share_bal))
    if sb_max > sb_min:
        sb_norm = (share_bal - sb_min) / (sb_max - sb_min + EPS)
        score = score + float(KNEE_BALANCE_SHARE_WEIGHT) * (sb_norm**2)

    score = score + float(KNEE_CAP_EXCESS_WEIGHT) * cap_excess
    score = score - float(KNEE_ISA_SHARE_WEIGHT) * isa_share
    score = score + float(KNEE_LAI_SHARE_EXCESS_WEIGHT) * np.maximum(lai_share - 0.45, 0.0) ** 2
    return int(idx_c[int(np.argmin(score))])


def find_matching_sol_id(sub: pd.DataFrame, row: pd.Series) -> int:
    if "sol_id" not in sub.columns:
        return -1
    diff = (
        (sub["y_pred_opt"].to_numpy(dtype=float) - float(row["y_pred_opt_best"])) ** 2
        + (sub["RR"].to_numpy(dtype=float) - float(row["RR_best"])) ** 2
        + (
            (
                sub["Cost_total"].to_numpy(dtype=float) - float(row["Cost_best"])
            )
            / max(abs(float(row["Cost_best"])), 1.0)
        )
        ** 2
    )
    return int(sub.iloc[int(np.argmin(diff))]["sol_id"])


def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, length=3.5, width=0.8)


def add_panel_label(ax, label: str, title: str):
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=FS_PANEL_LABEL,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.text(
        -0.005,
        1.06,
        title,
        transform=ax.transAxes,
        fontsize=FS_PANEL_TITLE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def barycentric_to_xy(isa: float, wtd: float, lai: float) -> tuple[float, float]:
    h = math.sqrt(3) / 2.0
    x = wtd + 0.5 * lai
    y = h * lai
    return x, y


def draw_ternary_axes(ax, text_scale: float = 1.0):
    h = math.sqrt(3) / 2.0
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
    ax.plot(tri[:, 0], tri[:, 1], color=GREY, lw=1.15)
    tick_vals = [0.2, 0.4, 0.6, 0.8]
    tick_labels_inc = {0.2: "4%", 0.4: "8%", 0.6: "12%", 0.8: "16%"}
    tick_labels_dec = {0.2: "16%", 0.4: "12%", 0.6: "8%", 0.8: "4%"}

    for v in tick_vals:
        p1 = barycentric_to_xy(v, 1 - v, 0)
        p2 = barycentric_to_xy(v, 0, 1 - v)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ECE7E2", lw=0.85, zorder=0)
        p1 = barycentric_to_xy(1 - v, v, 0)
        p2 = barycentric_to_xy(0, v, 1 - v)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ECE7E2", lw=0.85, zorder=0)
        p1 = barycentric_to_xy(1 - v, 0, v)
        p2 = barycentric_to_xy(0, 1 - v, v)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ECE7E2", lw=0.85, zorder=0)

    def tick_len_data(direction, mm=2.0):
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((d[0], d[1]))
        px_per_data = float(np.hypot(*(p1 - p0)))
        target_px = float(mm) / 25.4 * ax.figure.dpi
        return target_px / max(px_per_data, EPS)

    def draw_tick(point, normal, tick_dir, label=None, label_offset=0.047, ha="center", va="center"):
        p = np.asarray(point, dtype=float)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        d = np.asarray(tick_dir, dtype=float)
        d = d / np.linalg.norm(d)
        tick_len = tick_len_data(d, mm=2.0)
        p2 = p + tick_len * d
        ax.plot([p[0], p2[0]], [p[1], p2[1]], color=GREY, lw=0.95, solid_capstyle="round")
        if label is not None:
            t = p + label_offset * n
            ax.text(t[0], t[1], label, fontsize=(FS_SMALL + 2) * text_scale, color=GREY, ha=ha, va=va)

    bottom_normal = np.array([0.0, -1.0])
    left_normal = np.array([-h, 0.5])
    right_normal = np.array([h, 0.5])
    bottom_tick_dir = np.array([0.5, -h])
    left_tick_dir = np.array([-1.0, 0.0])
    right_tick_dir = np.array([0.5, h])

    for v in [0.8, 0.6, 0.4, 0.2]:
        draw_tick(
            barycentric_to_xy(v, 1 - v, 0),
            bottom_normal,
            bottom_tick_dir,
            label=tick_labels_dec[round(v, 1)],
            label_offset=0.043,
            ha="center",
            va="top",
        )

    for v in tick_vals:
        draw_tick(
            barycentric_to_xy(1 - v, 0, v),
            left_normal,
            left_tick_dir,
            label=tick_labels_inc[round(v, 1)],
            label_offset=0.050,
            ha="right",
            va="center",
        )
        draw_tick(
            barycentric_to_xy(0, v, 1 - v),
            right_normal,
            right_tick_dir,
            label=tick_labels_inc[round(v, 1)],
            label_offset=0.050,
            ha="left",
            va="center",
        )

    ax.text(0.50, -0.085, "ISA", fontsize=FS_AXIS * text_scale, ha="center", va="top")
    ax.text(0.04, 0.42, "LAI", fontsize=FS_AXIS * text_scale, rotation=60, rotation_mode="anchor", ha="center", va="center")
    ax.text(0.96, 0.42, "WTD", fontsize=FS_AXIS * text_scale, rotation=-60, rotation_mode="anchor", ha="center", va="center")

    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-0.22, h + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


def build_selected_solutions(pareto: pd.DataFrame, threshold: float, summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    records: list[dict] = []
    summary_by_row = None
    if summary is not None and len(summary):
        summary = summary.copy()
        if "status" in summary.columns:
            summary = summary[summary["status"].fillna("ok") == "ok"].copy()
        summary = summary.drop_duplicates(subset=["row_id"], keep="last").set_index("row_id", drop=False)
        summary_by_row = summary

    for row_id, sub in pareto.groupby("row_id", sort=False):
        sub = sub.copy().sort_values(["RR", "Cost_total"], ascending=[False, True]).reset_index(drop=True)
        sub["cost_m"] = sub["Cost_total"] / 1e6
        sub["isa_share"] = sub["Cost_ISA"] / np.maximum(sub["Cost_total"], 1e-9)
        sub["wt_share"] = sub["Cost_WT"] / np.maximum(sub["Cost_total"], 1e-9)
        sub["lai_share"] = sub["Cost_LAI"] / np.maximum(sub["Cost_total"], 1e-9)
        sub["efficiency_score"] = sub["RR"] / np.maximum(sub["cost_m"], 1e-6)

        rr_max = float(sub["RR"].max())
        mm_candidates = sub[sub["RR"] >= rr_max * float(MAX_MITIGATION_RR_BAND)].copy()
        if len(mm_candidates) == 0:
            mm_candidates = sub.copy()
        mm_wt_cut = float(mm_candidates["wt_share"].quantile(0.75))
        mm_candidates = mm_candidates[mm_candidates["wt_share"] >= mm_wt_cut].copy()
        if len(mm_candidates) == 0:
            mm_candidates = sub[sub["RR"] >= rr_max * float(MAX_MITIGATION_RR_BAND)].copy()
        maximum_mitigation = mm_candidates.sort_values(
            ["wt_share", "RR", "Cost_total"],
            ascending=[False, False, False],
        ).iloc[0]

        feasible = sub[sub["RR"] >= threshold]
        if len(feasible):
            target_first = feasible.sort_values(
                ["isa_share", "Cost_total", "RR"],
                ascending=[False, True, False],
            ).iloc[0]
        else:
            target_first = maximum_mitigation

        eff_max = float(sub["efficiency_score"].max())
        ce_candidates = sub[sub["efficiency_score"] >= eff_max * float(COST_EFFICIENCY_BAND)].copy()
        if len(ce_candidates) == 0:
            ce_candidates = sub.copy()
        ce_lai_cut = float(ce_candidates["lai_share"].quantile(0.75))
        ce_candidates = ce_candidates[ce_candidates["lai_share"] >= ce_lai_cut].copy()
        if len(ce_candidates) == 0:
            ce_candidates = sub[sub["efficiency_score"] >= eff_max * float(COST_EFFICIENCY_BAND)].copy()
        if len(ce_candidates) == 0:
            ce_candidates = sub.copy()
        cost_efficiency_first = ce_candidates.sort_values(
            ["lai_share", "efficiency_score", "Cost_total"],
            ascending=[False, False, True],
        ).iloc[0]

        rr = sub["RR"].to_numpy(dtype=float)
        cost = sub["cost_m"].to_numpy(dtype=float)
        rr_n = np.zeros_like(rr) if np.isclose(rr.max(), rr.min()) else (rr - rr.min()) / (rr.max() - rr.min())
        cost_n = np.zeros_like(cost) if np.isclose(cost.max(), cost.min()) else (cost - cost.min()) / (cost.max() - cost.min())
        balanced_score = np.sqrt((1.0 - rr_n) ** 2 + cost_n**2)
        balanced_compromise = sub.iloc[int(np.argmin(balanced_score))]

        if summary_by_row is not None and int(row_id) in summary_by_row.index:
            srow = summary_by_row.loc[int(row_id)]
            knee_compromise = {
                "source_row_id": int(srow["source_row_id"]),
                "sol_id": find_matching_sol_id(sub, srow),
                "NAME_EN_JX": str(srow["NAME_EN_JX"]),
                "Longitude": float(srow["Longitude"]),
                "Latitude": float(srow["Latitude"]),
                "y_pred_base": float(srow["y_pred_base"]),
                "y_pred_opt": float(srow["y_pred_opt_best"]),
                "RR": float(srow["RR_best"]),
                "Cost_total": float(srow["Cost_best"]),
                "Cost_ISA": float(srow["Cost_ISA"]),
                "Cost_WT": float(srow["Cost_WT"]),
                "Cost_LAI": float(srow["Cost_LAI"]),
                "sample_weight": float(srow["sample_weight"]),
            }
        else:
            knee_compromise = sub.iloc[pick_knee_solution_index(sub)]

        selected_by_strategy = {
            "达标优先": target_first,
            "极限减灾": maximum_mitigation,
            "性价比优先": cost_efficiency_first,
            "平衡折中": knee_compromise,
        }

        for strategy_cn, row in selected_by_strategy.items():
            records.append(
                {
                    "strategy": strategy_cn,
                    "row_id": int(row_id),
                    "source_row_id": int(row["source_row_id"]),
                    "sol_id": int(row["sol_id"]),
                    "province": str(row["NAME_EN_JX"]),
                    "Longitude": float(row["Longitude"]),
                    "Latitude": float(row["Latitude"]),
                    "y_pred_base": float(row["y_pred_base"]),
                    "y_pred_opt": float(row["y_pred_opt"]),
                    "RR": float(row["RR"]),
                    "Cost_total": float(row["Cost_total"]),
                    "Cost_ISA": float(row["Cost_ISA"]),
                    "Cost_WT": float(row["Cost_WT"]),
                    "Cost_LAI": float(row["Cost_LAI"]),
                    "sample_weight": float(row["sample_weight"]),
                }
            )

    selected = pd.DataFrame.from_records(records)
    selected["attain"] = selected["RR"] >= threshold
    return selected


def build_overall_summary(selected: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy_cn in ORDER_CN:
        sub = selected[selected["strategy"] == strategy_cn].copy()
        w = sub["sample_weight"]
        rows.append(
            {
                "strategy": strategy_cn,
                "n_grids": int(round(w.sum())),
                "attainment_rate_rr_ge_0_35": weighted_share(sub["RR"] >= threshold, w),
                "mean_RR": weighted_mean(sub["RR"], w),
                "median_RR": float(np.median(sub["RR"])),
                "mean_y_pred_base": weighted_mean(sub["y_pred_base"], w),
                "mean_y_pred_opt": weighted_mean(sub["y_pred_opt"], w),
                "mean_cost_total_million_usd": weighted_mean(sub["Cost_total"] / 1e6, w),
                "median_cost_total_million_usd": float(np.median(sub["Cost_total"] / 1e6)),
                "total_cost_billion_usd": float(np.sum(sub["Cost_total"] * w) / 1e9),
                "mean_cost_ISA_million_usd": weighted_mean(sub["Cost_ISA"] / 1e6, w),
                "mean_cost_WT_million_usd": weighted_mean(sub["Cost_WT"] / 1e6, w),
                "mean_cost_LAI_million_usd": weighted_mean(sub["Cost_LAI"] / 1e6, w),
            }
        )
    return pd.DataFrame(rows)


def build_province_summary(selected: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    for (strategy_cn, province), sub in selected.groupby(["strategy", "province"], dropna=True):
        w = sub["sample_weight"]
        rows.append(
            {
                "strategy": strategy_cn,
                "province": province,
                "n_grids": int(round(w.sum())),
                "attain_rate": weighted_share(sub["RR"] >= threshold, w),
                "mean_RR": weighted_mean(sub["RR"], w),
                "mean_cost_total_million_usd": weighted_mean(sub["Cost_total"] / 1e6, w),
            }
        )
    return pd.DataFrame(rows)


def build_combo_counts(selected: pd.DataFrame) -> pd.DataFrame:
    base = selected[["row_id", "strategy", "attain", "sample_weight"]].copy()
    att = (
        base.pivot(index="row_id", columns="strategy", values="attain")
        .fillna(False)
        .astype(bool)
    )
    weights = base.drop_duplicates("row_id").set_index("row_id")["sample_weight"]
    att = att[ORDER_CN]
    combo_counts = (
        att.assign(weight=weights)
        .groupby(ORDER_CN, dropna=False)["weight"]
        .sum()
        .reset_index(name="count_weight")
        .sort_values("count_weight", ascending=False)
        .reset_index(drop=True)
    )
    combo_counts["count"] = combo_counts["count_weight"].round().astype(int)
    combo_counts["fraction"] = combo_counts["count_weight"] / combo_counts["count_weight"].sum()

    def combo_to_label(row: pd.Series) -> str:
        labs = [MAP_EN[k] for k in ORDER_CN if bool(row[k])]
        return "None" if not labs else " + ".join(labs)

    combo_counts["label"] = combo_counts.apply(combo_to_label, axis=1)
    combo_counts["strategy_en_count"] = combo_counts[ORDER_CN].sum(axis=1)
    return combo_counts


def panel_a_strategy_landscape(ax, overall: pd.DataFrame):
    style_ax(ax)
    add_panel_label(ax, "a", "Strategy landscape")

    overall = overall.copy()
    overall["strategy_en"] = overall["strategy"].map(MAP_EN)
    overall = overall.set_index("strategy_en").loc[ORDER].reset_index()

    for _, row in overall.iterrows():
        ax.scatter(
            row["mean_cost_total_million_usd"],
            row["mean_RR"],
            s=2300 * row["attainment_rate_rr_ge_0_35"] + 170,
            color=COLORS[row["strategy_en"]],
            edgecolor="white",
            linewidth=1.7,
            zorder=3,
            alpha=0.98,
        )

    x_span = max(float(overall["mean_cost_total_million_usd"].max()), 1.0)
    y_min_data = float(overall["mean_RR"].min())
    y_max_data = float(overall["mean_RR"].max())
    y_span = max(y_max_data - y_min_data, 0.05)
    offsets = {
        "Target-first": (0.08 * x_span, -0.10 * y_span),
        "Maximum mitigation": (-0.14 * x_span, 0.10 * y_span),
        "Cost-efficiency-first": (0.12 * x_span, 0.06 * y_span),
        "Balanced compromise": (0.10 * x_span, -0.16 * y_span),
    }

    for _, row in overall.iterrows():
        dx, dy = offsets[row["strategy_en"]]
        label = (
            f"{row['strategy_en']}\n"
            f"Risk level change rate = {row['mean_RR']:.3f}\n"
            f"Mean cost per 100 km² = {row['mean_cost_total_million_usd']:.2f} M USD\n"
            f"Threshold-reaching area fraction = {row['attainment_rate_rr_ge_0_35']*100:.2f}%"
        )
        ax.annotate(
            label,
            xy=(row["mean_cost_total_million_usd"], row["mean_RR"]),
            xytext=(row["mean_cost_total_million_usd"] + dx, row["mean_RR"] + dy),
            fontsize=FS_ANNOT,
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LIGHT_GREY, lw=0.9),
            zorder=4,
        )

    tf = overall.loc[overall["strategy_en"] == "Target-first"].iloc[0]
    mm = overall.loc[overall["strategy_en"] == "Maximum mitigation"].iloc[0]
    cost_reduction_pct = (1.0 - tf["mean_cost_total_million_usd"] / mm["mean_cost_total_million_usd"]) * 100.0
    attain_gap_pct = abs(tf["attainment_rate_rr_ge_0_35"] - mm["attainment_rate_rr_ge_0_35"]) * 100.0
    if attain_gap_pct < 1e-6:
        compare_text = (
            "Target-first reaches the same threshold-oriented area fraction\n"
            f"as maximum mitigation, but at {cost_reduction_pct:.2f}% lower mean cost."
        )
    else:
        compare_text = (
            f"Target-first reduces mean cost by {cost_reduction_pct:.2f}% relative to\n"
            f"maximum mitigation, with a threshold-reaching area gap of {attain_gap_pct:.2f}%."
        )
    ax.text(
        0.50,
        0.98,
        compare_text,
        transform=ax.transAxes,
        fontsize=FS_SMALL,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=LIGHT_GREY, lw=0.9),
    )

    ax.set_xlabel("Mean cost per 100 km² in 20 years (million USD)", fontsize=FS_AXIS)
    ax.set_ylabel("Risk level change rate", fontsize=FS_AXIS)
    ax.set_xlim(0, float(overall["mean_cost_total_million_usd"].max()) * 1.25)
    ax.set_ylim(max(0.0, y_min_data - 0.05), min(1.0, y_max_data + 0.08))

    frac_values = sorted(set(np.quantile(overall["attainment_rate_rr_ge_0_35"], [0.2, 0.5, 0.8]).round(2)))
    legend_x, legend_y = 0.055, 0.88
    ax.text(
        legend_x,
        legend_y + 0.095,
        "Bubble area encodes the fraction of 100 km² areas\nthat reach the 0.35 threshold.",
        transform=ax.transAxes,
        fontsize=FS_SMALL,
        ha="left",
        va="bottom",
    )
    for i, frac in enumerate(frac_values[:3]):
        xpos = legend_x + i * 0.16
        ax.scatter(
            [xpos],
            [legend_y],
            s=2300 * frac + 170,
            color="none",
            edgecolor=GREY,
            linewidth=1.0,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            xpos,
            legend_y - 0.09,
            f"{frac*100:.0f}%",
            transform=ax.transAxes,
            fontsize=10.5,
            ha="center",
            va="top",
        )


def panel_b_area_ridgeline(ax, selected: pd.DataFrame):
    add_panel_label(ax, "b", "Area-level distributions")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, length=3.5, width=0.8)

    xs = np.linspace(0, 1, 400)
    base_levels = np.arange(len(ORDER))[::-1]
    for level, strategy_en in zip(base_levels, ORDER):
        strategy_cn = next(k for k, v in MAP_EN.items() if v == strategy_en)
        data = selected.loc[selected["strategy"] == strategy_cn, "RR"].to_numpy()
        kde = gaussian_kde(data, bw_method=0.08)
        ys = kde(xs)
        ys = ys / ys.max() * 0.84
        ax.fill_between(xs, level, level + ys, color=COLORS[strategy_en], alpha=0.78, linewidth=0)
        ax.plot(xs, level + ys, color=COLORS[strategy_en], lw=1.8)
        mean_v = data.mean()
        med_v = np.median(data)
        ax.scatter([mean_v], [level + 0.45], s=54, facecolor="white", edgecolor=GREY, linewidth=1.15, zorder=5)
        ax.scatter([med_v], [level + 0.45], s=44, facecolor=GREY, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(-0.035, level + 0.16, strategy_en, ha="right", va="center", fontsize=FS_SMALL)

    ax.axvline(THRESHOLD, color=GREY, lw=1.25, ls=(0, (5, 3)))
    ax.text(
        THRESHOLD,
        len(ORDER) - 0.13,
        f"Target threshold = {THRESHOLD:.2f}",
        fontsize=FS_SMALL,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=LIGHT_GREY, lw=0.8),
    )

    handles = [
        Line2D([0], [0], marker="o", markersize=7.5, markerfacecolor="white", markeredgecolor=GREY, linestyle="None", label="Mean"),
        Line2D([0], [0], marker="o", markersize=6.2, markerfacecolor=GREY, markeredgecolor="white", linestyle="None", label="Median"),
        Line2D([0], [0], color=GREY, lw=1.2, ls=(0, (5, 3)), label="Threshold"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=FS_LEGEND)
    ax.set_xlabel("Risk level change rate", fontsize=FS_AXIS)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.22, len(ORDER) + 0.25)
    ax.set_yticks([])


def panel_c_attainment_intersections(fig, gs_cell, combo_counts: pd.DataFrame):
    sub = gs_cell.subgridspec(2, 1, height_ratios=[2.25, 1.25], hspace=0.035)
    ax_bar = fig.add_subplot(sub[0])
    ax_mat = fig.add_subplot(sub[1], sharex=ax_bar)

    cc = combo_counts.copy()
    x = np.arange(len(cc))

    style_ax(ax_bar)
    add_panel_label(ax_bar, "c", "Threshold-reaching area intersections")
    bars = ax_bar.bar(x, cc["fraction"] * 100, color="#CDBFB5", edgecolor="white", linewidth=1.0)
    ax_bar.set_ylabel("Share of total area (%)", fontsize=FS_AXIS)
    ax_bar.set_xticks([])
    ax_bar.set_xlim(-0.6, len(cc) - 0.4)
    ax_bar.set_ylim(0, max(cc["fraction"] * 100) * 1.23)

    for bar, frac, count in zip(bars, cc["fraction"], cc["count"]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{frac*100:.2f}%\n(n={count:,})",
            ha="center",
            va="bottom",
            fontsize=10.3,
        )

    reached_any = float(cc.loc[cc["label"] != "None", "fraction"].sum() * 100)
    ax_bar.text(
        0.01,
        0.97,
        f"Only {reached_any:.2f}% of the analysed 100 km² areas reached the threshold\nunder at least one decision rule.",
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=FS_SMALL,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LIGHT_GREY, lw=0.8),
    )

    ax_mat.set_facecolor("white")
    for spine in ax_mat.spines.values():
        spine.set_visible(False)
    ax_mat.set_ylim(-0.5, len(ORDER) - 0.5)
    ax_mat.set_yticks(np.arange(len(ORDER)))
    ax_mat.set_yticklabels(ORDER, fontsize=FS_SMALL)
    ax_mat.invert_yaxis()
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels([""] * len(cc))
    ax_mat.tick_params(axis="x", length=0)

    for i in range(len(cc)):
        ys = []
        for j, strategy_cn in enumerate(ORDER_CN):
            strategy_en = MAP_EN[strategy_cn]
            active = bool(cc.loc[i, strategy_cn])
            if active:
                ax_mat.scatter(i, j, s=88, color=COLORS[strategy_en], edgecolor="white", linewidth=0.9, zorder=3)
                ys.append(j)
            else:
                ax_mat.scatter(i, j, s=30, color="#DDD7D1", edgecolor="none", zorder=2)
        if len(ys) >= 2:
            ax_mat.plot([i, i], [min(ys), max(ys)], color=GREY, lw=1.1, zorder=1)

    for i, lbl in enumerate(cc["label"]):
        ax_mat.text(i, len(ORDER) - 0.06, lbl.replace(" + ", "\n+\n"), ha="center", va="top", fontsize=9.2)


def panel_d_cost_ternary(ax, overall: pd.DataFrame):
    add_panel_label(ax, "d", "")
    d_text_scale = 0.8
    draw_ternary_axes(ax, text_scale=d_text_scale)

    overall = overall.copy()
    overall["strategy_en"] = overall["strategy"].map(MAP_EN)
    overall = overall.set_index("strategy_en").loc[ORDER].reset_index()

    for _, row in overall.iterrows():
        total = max(float(row["mean_cost_total_million_usd"]), 1e-9)
        isa = row["mean_cost_ISA_million_usd"] / total
        wtd = row["mean_cost_WT_million_usd"] / total
        lai = row["mean_cost_LAI_million_usd"] / total
        x, y = barycentric_to_xy(isa, wtd, lai)
        x_plot = x
        y_plot = y
        ax.scatter(x_plot, y_plot, s=150, color=COLORS[row["strategy_en"]], edgecolor="white", linewidth=1.5, zorder=3)
        text = f"{row['strategy_en']}\nISA {isa*100:.1f}% | WTD {wtd*100:.1f}% | LAI {lai*100:.1f}%"
        label_pos = {
            "Cost-efficiency-first": (0.26, 0.985, "center", "center"),
            "Balanced compromise": (0.49, 0.60, "center", "center"),
            "Target-first": (-0.30, 0.24, "left", "center"),
            "Maximum mitigation": (0.36, 0.14, "left", "center"),
        }[row["strategy_en"]]
        tx, ty, ha, va = label_pos
        ax.annotate(
            text,
            xy=(x_plot, y_plot),
            xytext=(tx, ty),
            fontsize=(FS_ANNOT - 0.3) * d_text_scale,
            ha=ha,
            va=va,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LIGHT_GREY, lw=0.9),
            annotation_clip=False,
        )

    return


def make_main_figure(save_path: Path, overall: pd.DataFrame, selected: pd.DataFrame, combo_counts: pd.DataFrame):
    fig = plt.figure(figsize=(12.4, 9.4), facecolor="white")
    outer = GridSpec(2, 2, figure=fig, left=0.048, right=0.992, top=0.94, bottom=0.055, wspace=0.23, hspace=0.23)
    axA = fig.add_subplot(outer[0, 0])
    axB = fig.add_subplot(outer[0, 1])
    panel_a_strategy_landscape(axA, overall)
    panel_b_area_ridgeline(axB, selected)
    panel_c_attainment_intersections(fig, outer[1, 0], combo_counts)
    axD = fig.add_subplot(outer[1, 1])
    panel_d_cost_ternary(axD, overall)
    fig.text(
        0.048,
        0.987,
        f"Fig. 1 | Comparative screening of four Pareto-based decision rules under a threshold of {THRESHOLD:.2f} in risk level change rate.",
        ha="left",
        va="top",
        fontsize=FS_FIG_TITLE,
        fontweight="bold",
    )
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def make_panel_A(save_path: Path, overall: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.3, 5.8), facecolor="white")
    panel_a_strategy_landscape(ax, overall)
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def make_panel_B(save_path: Path, selected: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.9, 5.8), facecolor="white")
    panel_b_area_ridgeline(ax, selected)
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def make_panel_C(save_path: Path, combo_counts: pd.DataFrame):
    fig = plt.figure(figsize=(7.9, 5.9), facecolor="white")
    gs = GridSpec(1, 1, figure=fig, left=0.11, right=0.985, top=0.92, bottom=0.12)
    panel_c_attainment_intersections(fig, gs[0, 0], combo_counts)
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def make_panel_D(save_path: Path, overall: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.9, 5.9), facecolor="white")
    panel_d_cost_ternary(ax, overall)
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def make_sup_province(save_path: Path, province: pd.DataFrame):
    prov = province.copy()
    prov["strategy_en"] = prov["strategy"].map(MAP_EN)
    prov = prov[prov["strategy_en"].isin(ORDER)]
    pivot = prov.pivot(index="province", columns="strategy_en", values="attain_rate")
    pivot = pivot.loc[pivot["Target-first"].sort_values(ascending=False).index]
    y = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(4.1, 9.2), facecolor="white")
    style_ax(ax)
    mins = pivot.min(axis=1).values * 100
    maxs = pivot.max(axis=1).values * 100
    ax.hlines(y, mins, maxs, color=LIGHT_GREY, lw=3.0, zorder=1)

    offsets = np.linspace(-0.18, 0.18, len(ORDER))
    for off, strategy_en in zip(offsets, ORDER):
        ax.scatter(
            pivot[strategy_en].values * 100,
            y + off,
            s=46,
            color=COLORS[strategy_en],
            edgecolor="white",
            linewidth=0.85,
            label=strategy_en,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index.tolist(), fontsize=10.2)
    ax.invert_yaxis()
    ax.set_xlabel("Threshold-reaching area fraction (%)", fontsize=FS_SMALL + 1)
    ax.set_xlim(0, 100)
    ax.set_title(
        "Supplementary Fig. 1 | Provincial heterogeneity in\nthreshold-reaching area fraction",
        fontsize=FS_SUP_TITLE,
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9.5, ncol=1, loc="lower right", handletextpad=0.4, borderpad=0.2)
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def write_markdown(save_path: Path, overall: pd.DataFrame, combo_counts: pd.DataFrame, selected: pd.DataFrame):
    tf = overall.loc[overall["strategy"] == "达标优先"].iloc[0]
    mm = overall.loc[overall["strategy"] == "极限减灾"].iloc[0]
    ce = overall.loc[overall["strategy"] == "性价比优先"].iloc[0]
    bc = overall.loc[overall["strategy"] == "平衡折中"].iloc[0]
    total_grids = int(overall["n_grids"].max())
    total_sampled_pareto = len(selected["row_id"].unique()) * int(round(selected["sample_weight"].iloc[0])) if "sample_weight" in selected else len(selected)
    reached_any = combo_counts.loc[combo_counts["label"] != "None", "count"].sum()
    tf_cost_reduction = (1.0 - tf["mean_cost_total_million_usd"] / mm["mean_cost_total_million_usd"]) * 100.0
    ce_lai = ce["mean_cost_LAI_million_usd"] / max(ce["mean_cost_total_million_usd"], 1e-9) * 100
    mm_wtd = mm["mean_cost_WT_million_usd"] / max(mm["mean_cost_total_million_usd"], 1e-9) * 100

    md_text = f"""# Figure legend

**Fig. 1 | Comparative screening of four Pareto-based decision rules under a threshold of {THRESHOLD:.2f} in risk level change rate.**  
**a,** Strategy landscape. The x axis shows **Mean cost per 100 km² in 20 years** and the y axis shows **Risk level change rate**. Bubble area is proportional to the fraction of 100 km² areas reaching the {THRESHOLD:.2f} threshold.  
**b,** Area-level ridgeline distributions of **Risk level change rate** for the four decision rules. The dashed vertical line marks the threshold of {THRESHOLD:.2f}. White dots indicate means and dark dots indicate medians.  
**c,** UpSet-style intersection analysis of threshold-reaching areas. Bars show the fraction of total analysed area belonging to each intersection, and the matrix below indicates which decision rules are involved.  
**d,** Ternary representation of cost composition among ISA, WTD and LAI. Point location indicates the relative contribution of the three cost components to the **Mean cost per 100 km² in 20 years**.  

Across {total_grids:,} analysed 100 km² areas and {total_sampled_pareto:,} sampled Pareto solutions, only {reached_any:,} areas ({reached_any / total_grids * 100:.2f}%) admitted at least one feasible solution reaching the {THRESHOLD:.2f} threshold in risk level change rate.

# Results

The policy-screening landscape shows a clear escalation from cost-efficiency-first to maximum mitigation in both expenditure and achieved reduction. Maximum mitigation yielded the highest mean risk level change rate ({mm['mean_RR']:.3f}), but it also imposed the largest **Mean cost per 100 km² in 20 years** ({mm['mean_cost_total_million_usd']:.2f} million USD). By contrast, target-first attained the same overall threshold-reaching area fraction as maximum mitigation ({tf['attainment_rate_rr_ge_0_35']*100:.2f}%) at a substantially lower mean cost ({tf['mean_cost_total_million_usd']:.2f} million USD), corresponding to a {tf_cost_reduction:.2f}% reduction in mean expenditure.

The full area-level distributions clarify why these aggregate differences emerge. Maximum mitigation shifts the entire distribution of risk level change rate upward, whereas cost-efficiency-first is concentrated near small improvements. Balanced compromise occupies an intermediate position, while target-first is concentrated around the threshold itself, reflecting its role as the post-Pareto selector used in the operational NSGA-II workflow.

The threshold-reaching intersection structure shows that the four rules do not simply differ in magnitude, but also in the sets of 100 km² areas they effectively serve. In particular, target-first and maximum mitigation share the same attainable area set under the current selection logic, implying that the major difference between them lies not in where the target can be reached, but in how aggressively they move along the attainable Pareto frontier once such areas are identified.

The ternary cost-composition view indicates that the strategies also differ mechanistically in how expenditure is allocated. Cost-efficiency-first is LAI-dominated, with LAI accounting for {ce_lai:.2f}% of its mean cost, whereas maximum mitigation shifts strongly toward WTD expenditure ({mm_wtd:.2f}% share).
"""
    save_path.write_text(md_text, encoding="utf-8")


def export_english_overall_csv(overall: pd.DataFrame, save_path: Path):
    overall = overall.copy()
    overall["strategy_en"] = overall["strategy"].map(MAP_EN)
    overall_en = overall.rename(
        columns={
            "strategy_en": "strategy",
            "attainment_rate_rr_ge_0_35": "threshold_reaching_area_fraction",
            "mean_RR": "mean_risk_level_change_rate",
            "median_RR": "median_risk_level_change_rate",
            "mean_y_pred_base": "mean_base_risk",
            "mean_y_pred_opt": "mean_optimised_risk",
            "mean_cost_total_million_usd": "mean_cost_per_100km2_in_20_years_million_usd",
            "median_cost_total_million_usd": "median_cost_per_100km2_in_20_years_million_usd",
            "total_cost_billion_usd": "total_cost_billion_usd",
            "mean_cost_ISA_million_usd": "mean_ISA_cost_million_usd",
            "mean_cost_WT_million_usd": "mean_WTD_cost_million_usd",
            "mean_cost_LAI_million_usd": "mean_LAI_cost_million_usd",
        }
    )
    keep_cols = [
        "strategy",
        "n_grids",
        "threshold_reaching_area_fraction",
        "mean_risk_level_change_rate",
        "median_risk_level_change_rate",
        "mean_base_risk",
        "mean_optimised_risk",
        "mean_cost_per_100km2_in_20_years_million_usd",
        "median_cost_per_100km2_in_20_years_million_usd",
        "total_cost_billion_usd",
        "mean_ISA_cost_million_usd",
        "mean_WTD_cost_million_usd",
        "mean_LAI_cost_million_usd",
    ]
    overall_en[keep_cols].to_csv(save_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto-csv", required=True, help="Path to nsga2_*_pareto.csv")
    parser.add_argument("--summary-csv", help="Optional path to nsga2_*_summary.csv for the RR_best / knee strategy")
    parser.add_argument("--out-dir", required=True, help="Base figure output directory")
    parser.add_argument("--tag", default="run", help="Subdirectory tag under out-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    pareto_csv = Path(args.pareto_csv).resolve()
    if args.summary_csv:
        summary_csv = Path(args.summary_csv).resolve()
    else:
        guess = Path(str(pareto_csv).replace("_pareto.csv", "_summary.csv"))
        summary_csv = guess if guess.exists() else None
    out_dir = Path(args.out_dir).resolve()
    save_dir = out_dir / f"nature_strategy_v5_{args.tag}"
    analysis_dir = save_dir / "analysis_inputs"
    save_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pareto = pd.read_csv(pareto_csv)
    required = {
        "row_id",
        "source_row_id",
        "sol_id",
        "Longitude",
        "Latitude",
        "NAME_EN_JX",
        "y_pred_base",
        "y_pred_opt",
        "RR",
        "Cost_total",
        "Cost_ISA",
        "Cost_WT",
        "Cost_LAI",
        "sample_weight",
    }
    missing = sorted(required - set(pareto.columns))
    if missing:
        raise ValueError(f"Missing required columns in pareto csv: {missing}")

    summary = None
    if summary_csv is not None:
        summary = pd.read_csv(summary_csv)

    selected = build_selected_solutions(pareto, THRESHOLD, summary=summary)
    overall = build_overall_summary(selected, THRESHOLD)
    province = build_province_summary(selected, THRESHOLD)
    combo_counts = build_combo_counts(selected)

    selected_csv = analysis_dir / "pareto_strategy_selected_solutions.csv"
    overall_csv = analysis_dir / "pareto_strategy_overall_summary.csv"
    province_csv = analysis_dir / "pareto_strategy_province_summary.csv"
    selected.to_csv(selected_csv, index=False)
    overall.to_csv(overall_csv, index=False)
    province.to_csv(province_csv, index=False)

    make_main_figure(save_dir / "nature_v5_main_figure.svg", overall, selected, combo_counts)
    make_panel_A(save_dir / "nature_v5_figA_strategy_landscape.svg", overall)
    make_panel_B(save_dir / "nature_v5_figB_area_ridgeline.svg", selected)
    make_panel_C(save_dir / "nature_v5_figC_attainment_intersections.svg", combo_counts)
    make_panel_D(save_dir / "nature_v5_figD_cost_ternary.svg", overall)
    make_sup_province(save_dir / "nature_v5_sup_fig1_province_rank_halfwidth.svg", province)
    write_markdown(save_dir / "nature_results_and_legends_v5.md", overall, combo_counts, selected)
    export_english_overall_csv(overall, save_dir / "nature_strategy_overall_summary_en_v5.csv")

    print("Saved nature-style outputs to:")
    print(save_dir)
    if summary_csv is not None:
        print(f"Summary strategy source: {summary_csv}")
    else:
        print("Summary strategy source: reconstructed from pareto fallback")
    for path in sorted(save_dir.glob("*.svg")):
        print(path)
    print("Saved analysis inputs to:")
    print(analysis_dir)
    for path in sorted(analysis_dir.glob("*.csv")):
        print(path)


if __name__ == "__main__":
    main()
