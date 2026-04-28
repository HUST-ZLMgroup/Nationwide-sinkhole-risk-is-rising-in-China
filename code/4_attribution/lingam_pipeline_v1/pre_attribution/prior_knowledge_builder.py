from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lingam_pipeline_v1.pre_attribution.attribution_config import ensure_common_output_dir
from lingam_pipeline_v1.pre_attribution.variable_schema import build_variable_dictionary, node_order


HYDROGEOLOGY_ROOTS = {"DK", "DB", "DF", "HDS"}
CLIMATE_ROOTS = {"PR", "TAS", "HUSS"}
HUMAN_ROOTS = {"PT"}
TARGET_NODE = "RISK"

REQUIRED_PATHS = {
    ("PT", "UF"),
    ("PT", "IP"),
    ("UF", "IP"),
    ("WTD", "LAI"),
}

FORBIDDEN_PATHS = {
    ("UF", "PT"),
    ("IP", "PT"),
    ("IP", "UF"),
    ("LAI", "WTD"),
}

LINGAM_VALUE_MAP = {
    "unconstrained": -1,
    "forbidden": 0,
    "required_path": 1,
    "diagonal": -1,
}


def classify_constraint(source: str, target: str) -> tuple[str, str]:
    if source == target:
        return "diagonal", "Self-loop is excluded from causal discovery."

    if source == TARGET_NODE:
        return "forbidden", "The target node is a sink and cannot point to any driver."

    if target in HYDROGEOLOGY_ROOTS:
        return "forbidden", "Static hydrogeology is treated as exogenous and receives no incoming paths."

    if target in CLIMATE_ROOTS:
        return "forbidden", "Climate forcing is treated as exogenous and receives no incoming paths."

    if target in HUMAN_ROOTS:
        return "forbidden", "Population total is treated as a human-pressure root node."

    if (source, target) in REQUIRED_PATHS:
        return "required_path", "This direction is imposed as a strong prior path."

    if (source, target) in FORBIDDEN_PATHS:
        return "forbidden", "This reverse direction is rejected by prior causal ordering."

    return "unconstrained", "No hard prior is imposed on this ordered pair."


def build_prior_knowledge_edges() -> pd.DataFrame:
    dictionary = build_variable_dictionary().set_index("node")
    nodes = node_order()
    rows = []

    for target_idx, target in enumerate(nodes):
        for source_idx, source in enumerate(nodes):
            constraint_type, reason = classify_constraint(source, target)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "source_group": dictionary.loc[source, "group_lv1"],
                    "target_group": dictionary.loc[target, "group_lv1"],
                    "constraint_type": constraint_type,
                    "lingam_value": LINGAM_VALUE_MAP[constraint_type],
                    "matrix_row_index": target_idx,
                    "matrix_col_index": source_idx,
                    "reason": reason,
                }
            )
    return pd.DataFrame.from_records(rows)


def build_prior_knowledge_matrix(edges_df: pd.DataFrame) -> pd.DataFrame:
    nodes = node_order()
    matrix = np.full((len(nodes), len(nodes)), -1, dtype=int)

    for row in edges_df.itertuples(index=False):
        matrix[row.matrix_row_index, row.matrix_col_index] = int(row.lingam_value)

    np.fill_diagonal(matrix, -1)
    return pd.DataFrame(matrix, index=nodes, columns=nodes)


def write_prior_knowledge(output_dir: str | Path | None = None) -> dict[str, Path]:
    out_dir = ensure_common_output_dir(output_dir)
    edges_df = build_prior_knowledge_edges()
    matrix_df = build_prior_knowledge_matrix(edges_df)
    nodes = node_order()

    edges_path = out_dir / "prior_knowledge_edges.csv"
    required_path = out_dir / "prior_knowledge_required_paths.csv"
    forbidden_path = out_dir / "prior_knowledge_forbidden_paths.csv"
    matrix_path = out_dir / "prior_knowledge_matrix.csv"
    matrix_nodes_path = out_dir / "prior_knowledge_matrix_nodes.csv"
    metadata_path = out_dir / "prior_knowledge_metadata.json"

    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    edges_df.loc[edges_df["constraint_type"] == "required_path"].to_csv(
        required_path, index=False, encoding="utf-8-sig"
    )
    edges_df.loc[edges_df["constraint_type"] == "forbidden"].to_csv(
        forbidden_path, index=False, encoding="utf-8-sig"
    )
    matrix_df.to_csv(matrix_path, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "node": nodes,
            "matrix_order": list(range(len(nodes))),
        }
    ).to_csv(matrix_nodes_path, index=False, encoding="utf-8-sig")

    metadata = {
        "matrix_semantics": {
            "row_axis": "target node",
            "column_axis": "source node",
            "lingam_values": {
                "-1": "unconstrained / unknown",
                "0": "no directed path from source to target",
                "1": "directed path required from source to target",
            },
        },
        "nodes": nodes,
        "summary": {
            "n_nodes": len(nodes),
            "n_forbidden": int((edges_df["constraint_type"] == "forbidden").sum()),
            "n_required_path": int((edges_df["constraint_type"] == "required_path").sum()),
            "n_unconstrained": int((edges_df["constraint_type"] == "unconstrained").sum()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "edges": edges_path,
        "required": required_path,
        "forbidden": forbidden_path,
        "matrix": matrix_path,
        "matrix_nodes": matrix_nodes_path,
        "metadata": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prior knowledge files for LiNGAM.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prior knowledge outputs. Defaults to outputs/common.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_prior_knowledge(args.output_dir)
    print(f"[OK] prior_knowledge_edges.csv -> {paths['edges']}")
    print(f"[OK] prior_knowledge_required_paths.csv -> {paths['required']}")
    print(f"[OK] prior_knowledge_forbidden_paths.csv -> {paths['forbidden']}")
    print(f"[OK] prior_knowledge_matrix.csv -> {paths['matrix']}")
    print(f"[OK] prior_knowledge_matrix_nodes.csv -> {paths['matrix_nodes']}")
    print(f"[OK] prior_knowledge_metadata.json -> {paths['metadata']}")


if __name__ == "__main__":
    main()

