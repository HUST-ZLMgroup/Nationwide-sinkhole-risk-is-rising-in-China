# -*- coding: utf-8 -*-
"""pearson_spearman_code_v6_12vars_twopanel_editable_svg.py

Pearson / Spearman 相关性网络图（12变量版）
本版按用户给定参考图继续微调：
- 保留 Pearson 与 Spearman
- 保留节点标签、子图标题、面板字母、色条刻度与色条标题
- 不恢复冗余图例
- 线宽整体减薄，弱化厚重感
- 将不显著边画得更细、更浅
- 保持新增 TAS 与 HUSS
- 输出单图与双图合并版
- SVG 文本保持为可编辑文本，不转曲、不嵌入字体
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.unicode_minus"] = False

try:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
except Exception:
    pass

COLOR_SCHEMES = {
    1: {"nodes": plt.cm.RdBu_r, "edges": plt.cm.PRGn},
}

CORRELATION_METHODS = {"pearson", "spearman"}


@dataclass
class CorrResult:
    features: List[str]
    corr_target: np.ndarray
    p_target: np.ndarray
    corr_matrix: np.ndarray
    p_matrix: np.ndarray


def calculate_corr_p(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    method = method.lower().strip()
    if method == "pearson":
        return pearsonr(x, y)
    if method == "spearman":
        return spearmanr(x, y)
    raise ValueError(f"Unknown method: {method}")


def compute_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    method: str,
) -> CorrResult:
    use_cols = feature_cols + [target_col]
    sub = df[use_cols].copy()
    sub = sub.apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(axis=0, how="any")

    y = sub[target_col].values
    X = sub[feature_cols].values
    n_feat = len(feature_cols)

    corr_target = np.zeros(n_feat, dtype=float)
    p_target = np.zeros(n_feat, dtype=float)
    for i in range(n_feat):
        r, p = calculate_corr_p(X[:, i], y, method)
        corr_target[i] = r
        p_target[i] = p

    corr_matrix = np.zeros((n_feat, n_feat), dtype=float)
    p_matrix = np.ones((n_feat, n_feat), dtype=float)
    for i in range(n_feat):
        corr_matrix[i, i] = 1.0
        p_matrix[i, i] = 0.0
        for j in range(i + 1, n_feat):
            r, p = calculate_corr_p(X[:, i], X[:, j], method)
            corr_matrix[i, j] = corr_matrix[j, i] = r
            p_matrix[i, j] = p_matrix[j, i] = p

    return CorrResult(
        features=list(feature_cols),
        corr_target=corr_target,
        p_target=p_target,
        corr_matrix=corr_matrix,
        p_matrix=p_matrix,
    )


def summarize_results(
    features: List[str],
    corr_target: np.ndarray,
    p_target: np.ndarray,
    corr_matrix: np.ndarray,
    p_matrix: np.ndarray,
    method_name: str,
    out_dir: str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    df_importance = (
        pd.DataFrame(
            {
                "Feature": features,
                "Corr_abs": np.abs(corr_target),
                "Corr_signed": corr_target,
                "P_value": p_target,
                "Significance": ["**" if p < 0.01 else "*" if p < 0.05 else "-" for p in p_target],
            }
        )
        .sort_values("Corr_abs", ascending=False)
    )

    rows = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "Feature_1": features[i],
                    "Feature_2": features[j],
                    "Corr_abs": abs(float(corr_matrix[i, j])),
                    "Corr_signed": float(corr_matrix[i, j]),
                    "P_value": float(p_matrix[i, j]),
                    "Significant(p<0.05)": "Yes" if float(p_matrix[i, j]) < 0.05 else "No",
                }
            )
    df_interactions = pd.DataFrame(rows).sort_values("Corr_abs", ascending=False)

    path1 = os.path.join(out_dir, f"corr_table_feature_vs_y_{method_name}.csv")
    path2 = os.path.join(out_dir, f"corr_table_feature_vs_feature_{method_name}.csv")
    df_importance.to_csv(path1, index=False, encoding="utf-8-sig")
    df_interactions.to_csv(path2, index=False, encoding="utf-8-sig")
    return path1, path2


def _circle_pos(features: List[str], radius: float = 1.0) -> Dict[str, Tuple[float, float]]:
    n = len(features)
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, n, endpoint=False)
    return {f: (radius * np.cos(a), radius * np.sin(a)) for f, a in zip(features, angles)}


def _nice_ticks(vmin: float, vmax: float, n: int = 7) -> List[float]:
    if np.isclose(vmin, vmax):
        return [round(vmin, 2)]
    vals = np.linspace(vmin, vmax, n)
    return [round(float(v), 1) for v in vals]


def _draw_one_panel(
    fig,
    panel_rect: List[float],
    features: List[str],
    corr_target: np.ndarray,
    p_target: np.ndarray,
    corr_matrix: np.ndarray,
    p_matrix: np.ndarray,
    cmap_nodes,
    cmap_edges,
    title_text: str,
    panel_letter: Optional[str] = None,
    show_labels: bool = True,
    title_fs: float = 8.8,
    label_fs: float = 8.4,
    tick_fs: float = 7.2,
):
    pl, pb, pw, ph = panel_rect

    ax = fig.add_axes([pl, pb, pw * 0.79, ph], aspect="equal")
    ax.set_axis_off()

    G = nx.Graph()
    G.add_nodes_from(features)
    pos = _circle_pos(features, radius=1.0)

    interaction_abs = np.abs(corr_matrix.copy())
    np.fill_diagonal(interaction_abs, 0.0)
    interaction_signed = corr_matrix.copy()
    np.fill_diagonal(interaction_signed, 0.0)

    edge_vmin = float(np.nanmin(interaction_signed))
    edge_vmax = float(np.nanmax(interaction_signed))
    node_vmin = float(np.nanmin(corr_target))
    node_vmax = float(np.nanmax(corr_target))
    norm_edges = mcolors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    norm_nodes = mcolors.Normalize(vmin=node_vmin, vmax=node_vmax)

    max_interaction_abs = float(np.nanmax(interaction_abs)) if np.nanmax(interaction_abs) > 0 else 1.0

    interactions = []
    n_features = len(features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            strength_abs = float(interaction_abs[i, j])
            strength_signed = float(interaction_signed[i, j])
            p_val = float(p_matrix[i, j])
            if strength_abs > 0:
                interactions.append((features[i], features[j], strength_abs, strength_signed, p_val))
    interactions.sort(key=lambda x: x[2])

    for u, v, strength_abs, strength_signed, p_val in interactions:
        color = cmap_edges(norm_edges(strength_signed))
        base_w = 0.14 + (strength_abs / max_interaction_abs) * 2.25
        if p_val < 0.05:
            width = base_w
            alpha = 0.82
            linestyle = "-"
        else:
            width = base_w * 0.72
            alpha = 0.24
            linestyle = (0, (1.6, 1.6))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=[color],
            style=linestyle,
            alpha=alpha,
            ax=ax,
        )

    imp_abs = np.abs(corr_target)
    size_min, size_max = 120.0, 255.0
    if float(np.nanmax(imp_abs)) - float(np.nanmin(imp_abs)) < 1e-12:
        node_sizes = np.full_like(imp_abs, (size_min + size_max) / 2.0, dtype=float)
    else:
        node_sizes = size_min + (imp_abs - float(np.nanmin(imp_abs))) / (
            float(np.nanmax(imp_abs)) - float(np.nanmin(imp_abs))
        ) * (size_max - size_min)

    node_colors = [cmap_nodes(norm_nodes(v)) for v in corr_target]
    node_edge_colors = ["black" if float(p) < 0.05 else "#7a7a7a" for p in p_target]
    node_line_widths = [0.9 if float(p) < 0.05 else 0.5 for p in p_target]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=node_line_widths,
        node_shape="o",
        ax=ax,
    )

    if show_labels:
        for node, (x, y) in pos.items():
            lx, ly = x * 1.15, y * 1.15
            ha = "center"
            if lx > 0.08:
                ha = "left"
            elif lx < -0.08:
                ha = "right"
            ax.text(lx, ly, node, fontsize=label_fs, ha=ha, va="center")

    ax.set_xlim(-1.24, 1.24)
    ax.set_ylim(-1.24, 1.24)
    ax.set_title(title_text, fontsize=title_fs, pad=8)

    if panel_letter:
        ax.text(-1.46, 1.40, panel_letter, fontsize=12.5, fontweight="bold", ha="left", va="top")

    # 色条 1：feature-feature
    cax1 = fig.add_axes([pl + pw * 0.83, pb + ph * 0.56, pw * 0.028, ph * 0.34])
    sm1 = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges)
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, cax=cax1)
    cb1.outline.set_visible(False)
    cb1.set_ticks(_nice_ticks(edge_vmin, edge_vmax, n=7))
    cb1.ax.tick_params(labelsize=tick_fs, length=2.0, width=0.45, pad=1.0)
    cb1.set_label("Between features", rotation=270, labelpad=8, fontsize=7.6)

    # 色条 2：feature-disaster
    cax2 = fig.add_axes([pl + pw * 0.83, pb + ph * 0.08, pw * 0.028, ph * 0.34])
    sm2 = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, cax=cax2)
    cb2.outline.set_visible(False)
    cb2.set_ticks(_nice_ticks(node_vmin, node_vmax, n=7))
    cb2.ax.tick_params(labelsize=tick_fs, length=2.0, width=0.45, pad=1.0)
    cb2.set_label("Feature-disaster", rotation=270, labelpad=8, fontsize=7.6)

    return ax, cax1, cax2


def plot_one_method(
    features: List[str],
    corr_target: np.ndarray,
    p_target: np.ndarray,
    corr_matrix: np.ndarray,
    p_matrix: np.ndarray,
    method_name: str,
    out_dir: str,
    fig_w_cm: float = 8.8,
    fig_h_cm: float = 6.6,
    scheme_index: int = 1,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54))
    current_color_scheme = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])

    title_map = {
        "pearson": "Correlation Network (Pearson)",
        "spearman": "Correlation Network (Spearman)",
    }

    _draw_one_panel(
        fig=fig,
        panel_rect=[0.05, 0.08, 0.90, 0.84],
        features=features,
        corr_target=corr_target,
        p_target=p_target,
        corr_matrix=corr_matrix,
        p_matrix=p_matrix,
        cmap_nodes=current_color_scheme["nodes"],
        cmap_edges=current_color_scheme["edges"],
        title_text=title_map[method_name],
        panel_letter=None,
        show_labels=True,
    )

    svg_path = os.path.join(out_dir, f"corr_network_{method_name}.svg")
    png_path = os.path.join(out_dir, f"corr_network_{method_name}.png")
    fig.savefig(svg_path, dpi=300, transparent=True)
    fig.savefig(png_path, dpi=300, transparent=True)
    plt.close(fig)
    return svg_path, png_path


def plot_two_panel(
    results: Dict[str, CorrResult],
    plot_features: List[str],
    out_dir: str,
    fig_w_cm: float = 18.4,
    fig_h_cm: float = 6.9,
    scheme_index: int = 1,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54))
    current_color_scheme = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])

    _draw_one_panel(
        fig=fig,
        panel_rect=[0.03, 0.10, 0.43, 0.80],
        features=plot_features,
        corr_target=results["pearson"].corr_target,
        p_target=results["pearson"].p_target,
        corr_matrix=results["pearson"].corr_matrix,
        p_matrix=results["pearson"].p_matrix,
        cmap_nodes=current_color_scheme["nodes"],
        cmap_edges=current_color_scheme["edges"],
        title_text="Correlation Network (Pearson)",
        panel_letter="b",
        show_labels=True,
    )
    _draw_one_panel(
        fig=fig,
        panel_rect=[0.51, 0.10, 0.43, 0.80],
        features=plot_features,
        corr_target=results["spearman"].corr_target,
        p_target=results["spearman"].p_target,
        corr_matrix=results["spearman"].corr_matrix,
        p_matrix=results["spearman"].p_matrix,
        cmap_nodes=current_color_scheme["nodes"],
        cmap_edges=current_color_scheme["edges"],
        title_text="Correlation Network (Spearman)",
        panel_letter="c",
        show_labels=True,
    )

    svg_path = os.path.join(out_dir, "corr_network_pearson_spearman_twopanel_v6_editable_text.svg")
    png_path = os.path.join(out_dir, "corr_network_pearson_spearman_twopanel_v6_editable_text.png")
    fig.savefig(svg_path, dpi=300, transparent=True)
    fig.savefig(png_path, dpi=300, transparent=True)
    plt.close(fig)
    return svg_path, png_path


def run(
    data_path: str,
    out_dir: str,
    feature_cols: List[str],
    target_col: str,
    feature_rename_map: Optional[Dict[str, str]] = None,
    methods: Iterable[str] = ("pearson", "spearman"),
    scheme_index: int = 1,
    fig_w_cm_single: float = 8.8,
    fig_h_cm_single: float = 6.6,
    fig_w_cm_combo: float = 18.2,
    fig_h_cm_combo: float = 6.9,
) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(data_path)

    if feature_rename_map is None:
        plot_features = feature_cols
    else:
        plot_features = [feature_rename_map.get(c, c) for c in feature_cols]

    results: Dict[str, CorrResult] = {}
    output_paths: Dict[str, Dict[str, str]] = {}

    for method in methods:
        method = method.lower().strip()
        if method not in CORRELATION_METHODS:
            raise ValueError(f"Invalid method: {method}. Choose from {sorted(CORRELATION_METHODS)}")

        corr_res = compute_correlations(df, feature_cols, target_col, method)
        results[method] = CorrResult(
            features=plot_features,
            corr_target=corr_res.corr_target,
            p_target=corr_res.p_target,
            corr_matrix=corr_res.corr_matrix,
            p_matrix=corr_res.p_matrix,
        )

        method_out_dir = os.path.join(out_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        table1, table2 = summarize_results(
            plot_features,
            corr_res.corr_target,
            corr_res.p_target,
            corr_res.corr_matrix,
            corr_res.p_matrix,
            method,
            method_out_dir,
        )
        svg1, png1 = plot_one_method(
            plot_features,
            corr_res.corr_target,
            corr_res.p_target,
            corr_res.corr_matrix,
            corr_res.p_matrix,
            method,
            method_out_dir,
            fig_w_cm=fig_w_cm_single,
            fig_h_cm=fig_h_cm_single,
            scheme_index=scheme_index,
        )
        output_paths[method] = {
            "svg": svg1,
            "png": png1,
            "table_feature_y": table1,
            "table_feature_feature": table2,
        }

    combo_svg, combo_png = plot_two_panel(
        results=results,
        plot_features=plot_features,
        out_dir=out_dir,
        fig_w_cm=fig_w_cm_combo,
        fig_h_cm=fig_h_cm_combo,
        scheme_index=scheme_index,
    )
    output_paths["combined"] = {"svg": combo_svg, "png": combo_png}
    return output_paths


def _parse_methods(m: str) -> List[str]:
    if not m:
        return ["pearson", "spearman"]
    if "," in m:
        return [x.strip() for x in m.split(",") if x.strip()]
    return [m.strip()]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Draw Pearson / Spearman correlation network plots.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="Disaster")
    parser.add_argument("--methods", type=str, default="pearson,spearman")
    parser.add_argument("--scheme_index", type=int, default=1)
    parser.add_argument("--fig_w_cm_single", type=float, default=8.8)
    parser.add_argument("--fig_h_cm_single", type=float, default=6.6)
    parser.add_argument("--fig_w_cm_combo", type=float, default=18.2)
    parser.add_argument("--fig_h_cm_combo", type=float, default=6.9)
    args = parser.parse_args()

    # 顺序按参考图调整，并把 TAS/HUSS 插入 PR 与 WTD 附近
    feature_cols = [
        "UrbanFrac_hist_2000_2010_2020",
        "Distance_to_Fault_m",
        "Depth_to_Bedrock",
        "Distance_to_karst",
        "HDS_hist_2000_2010_2020",
        "WTD_hist_2000_2010_2020",
        "Huss_hist_2000_2010_2020",
        "Tas_hist_2000_2010_2020",
        "Precip_hist_2000_2010_2020",
        "LAI_hist_2000_2010_2020",
        "ImperviousIndex_hist_2000_2010_2020",
        "PopTotal_hist_2000_2010_2020",
    ]

    feature_rename_map = {
        "Distance_to_karst": "DK",
        "Depth_to_Bedrock": "DB",
        "Distance_to_Fault_m": "DF",
        "UrbanFrac_hist_2000_2010_2020": "UF",
        "PopTotal_hist_2000_2010_2020": "PT",
        "ImperviousIndex_hist_2000_2010_2020": "IP",
        "LAI_hist_2000_2010_2020": "LAI",
        "Precip_hist_2000_2010_2020": "PR",
        "Tas_hist_2000_2010_2020": "TAS",
        "Huss_hist_2000_2010_2020": "HUSS",
        "WTD_hist_2000_2010_2020": "WTD",
        "HDS_hist_2000_2010_2020": "HDS",
    }

    methods = _parse_methods(args.methods)
    run(
        data_path=args.data_path,
        out_dir=args.out_dir,
        feature_cols=feature_cols,
        target_col=args.target_col,
        feature_rename_map=feature_rename_map,
        methods=methods,
        scheme_index=args.scheme_index,
        fig_w_cm_single=args.fig_w_cm_single,
        fig_h_cm_single=args.fig_h_cm_single,
        fig_w_cm_combo=args.fig_w_cm_combo,
        fig_h_cm_combo=args.fig_h_cm_combo,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
