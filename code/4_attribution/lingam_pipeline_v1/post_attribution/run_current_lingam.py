from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lingam import DirectLiNGAM
from sklearn.preprocessing import StandardScaler

from lingam_pipeline_v1.pre_attribution.attribution_config import OUTPUT_ROOT
from lingam_pipeline_v1.pre_attribution.variable_schema import build_variable_dictionary, node_order


DEFAULT_CURRENT_CSV = OUTPUT_ROOT / "current" / "current_samples_for_lingam.csv"
DEFAULT_PRIOR_MATRIX = OUTPUT_ROOT / "common" / "prior_knowledge_matrix.csv"


def _load_prior_matrix(prior_matrix_csv: str | Path, nodes: list[str]) -> np.ndarray:
    prior_df = pd.read_csv(prior_matrix_csv, index_col=0)
    if prior_df.index.tolist() != nodes or prior_df.columns.tolist() != nodes:
        prior_df = prior_df.reindex(index=nodes, columns=nodes)
    if prior_df.isna().any().any():
        raise ValueError("prior_knowledge_matrix.csv has missing values after node reindexing.")
    return prior_df.to_numpy(dtype=int)


def _fit_current_lingam(current_csv: str | Path, prior_matrix_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    dictionary = build_variable_dictionary().set_index("node")
    nodes = node_order()
    df = pd.read_csv(current_csv)

    missing = [col for col in nodes if col not in df.columns]
    if missing:
        raise KeyError(f"current_samples_for_lingam.csv is missing node columns: {missing}")

    X = df.loc[:, nodes].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("Current LiNGAM input contains NaN values.")

    prior_knowledge = _load_prior_matrix(prior_matrix_csv, nodes)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    model = DirectLiNGAM(random_state=0, prior_knowledge=prior_knowledge)
    model.fit(Xz)

    B = np.asarray(model.adjacency_matrix_, dtype=float)
    W = np.linalg.inv(np.eye(B.shape[0]) - B)
    risk_idx = nodes.index("RISK")
    total_to_risk = W[risk_idx, :].copy()
    total_to_risk[risk_idx] = 0.0

    edge_rows = []
    for target_idx, target in enumerate(nodes):
        for source_idx, source in enumerate(nodes):
            if source == target:
                continue
            weight = float(B[target_idx, source_idx])
            if np.isclose(weight, 0.0, atol=1e-12):
                continue
            edge_rows.append(
                {
                    "source": source,
                    "target": target,
                    "source_group": dictionary.loc[source, "group_lv1"],
                    "target_group": dictionary.loc[target, "group_lv1"],
                    "edge_weight": weight,
                    "edge_weight_abs": abs(weight),
                    "edge_sign": "positive" if weight > 0 else "negative",
                    "is_direct_to_target": bool(target == "RISK"),
                    "is_mediator_edge": bool(target != "RISK" and source != "RISK"),
                    "is_selected_for_main_figure": False,
                }
            )

    edges_df = pd.DataFrame(edge_rows).sort_values(
        by=["is_direct_to_target", "edge_weight_abs"], ascending=[False, False]
    ).reset_index(drop=True)
    if not edges_df.empty:
        top_idx = edges_df["edge_weight_abs"].nlargest(min(12, len(edges_df))).index
        edges_df.loc[top_idx, "is_selected_for_main_figure"] = True

    total_rows = []
    for node in nodes:
        if node == "RISK":
            continue
        effect = float(total_to_risk[nodes.index(node)])
        total_rows.append(
            {
                "node": node,
                "node_full_name": dictionary.loc[node, "node_full_name"],
                "group_lv1": dictionary.loc[node, "group_lv1"],
                "group_lv2": dictionary.loc[node, "group_lv2"],
                "total_effect": effect,
                "total_effect_abs": abs(effect),
                "effect_sign": "positive" if effect > 0 else ("negative" if effect < 0 else "zero"),
            }
        )

    totals_df = pd.DataFrame(total_rows).sort_values("total_effect_abs", ascending=False).reset_index(drop=True)
    totals_df["rank_abs"] = np.arange(1, len(totals_df) + 1)
    totals_df["selected_topk"] = totals_df["rank_abs"] <= min(6, len(totals_df))

    metadata = {
        "n_samples": int(len(df)),
        "node_order": nodes,
        "target_node": "RISK",
        "target_definition": "current observed Disaster mapped to canonical RISK node",
        "causal_order": [nodes[i] for i in getattr(model, "causal_order_", [])],
        "n_nonzero_direct_edges": int(len(edges_df)),
    }
    return edges_df, totals_df, B, metadata


def write_current_lingam_outputs(
    current_csv: str | Path,
    prior_matrix_csv: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / "current"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_df, totals_df, B, metadata = _fit_current_lingam(current_csv, prior_matrix_csv)
    edges_path = out_dir / "current_lingam_edges_long.csv"
    totals_path = out_dir / "current_total_effects_to_target.csv"
    adjacency_path = out_dir / "current_adjacency_matrix_B.csv"
    metadata_path = out_dir / "current_lingam_metadata.json"

    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    totals_df.to_csv(totals_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(B, index=node_order(), columns=node_order()).to_csv(adjacency_path, encoding="utf-8-sig")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "edges": edges_path,
        "totals": totals_path,
        "adjacency": adjacency_path,
        "metadata": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a current-stage LiNGAM fit once.")
    parser.add_argument(
        "--current-csv",
        type=Path,
        default=DEFAULT_CURRENT_CSV,
        help="Current-stage sample table produced by build_current_samples_for_lingam.py",
    )
    parser.add_argument(
        "--prior-matrix-csv",
        type=Path,
        default=DEFAULT_PRIOR_MATRIX,
        help="Prior knowledge matrix CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "current",
        help="Directory for current-stage LiNGAM outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_current_lingam_outputs(args.current_csv, args.prior_matrix_csv, args.output_dir)
    print(f"[OK] current_lingam_edges_long.csv -> {paths['edges']}")
    print(f"[OK] current_total_effects_to_target.csv -> {paths['totals']}")
    print(f"[OK] current_adjacency_matrix_B.csv -> {paths['adjacency']}")
    print(f"[OK] current_lingam_metadata.json -> {paths['metadata']}")


if __name__ == "__main__":
    main()
