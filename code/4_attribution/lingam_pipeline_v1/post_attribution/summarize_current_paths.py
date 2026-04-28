from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from lingam_pipeline_v1.pre_attribution.attribution_config import OUTPUT_ROOT
from lingam_pipeline_v1.pre_attribution.variable_schema import build_variable_dictionary


DEFAULT_ADJACENCY_CSV = OUTPUT_ROOT / "current" / "current_adjacency_matrix_B.csv"
DEFAULT_TOTAL_EFFECT_CSV = OUTPUT_ROOT / "current" / "current_total_effects_to_target.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "current"
TARGET_NODE = "RISK"


def _load_adjacency(adjacency_csv: str | Path) -> pd.DataFrame:
    adjacency_df = pd.read_csv(adjacency_csv, index_col=0)
    if TARGET_NODE not in adjacency_df.index or TARGET_NODE not in adjacency_df.columns:
        raise KeyError(f"Adjacency matrix must contain the target node: {TARGET_NODE}")
    if adjacency_df.index.tolist() != adjacency_df.columns.tolist():
        raise ValueError("Adjacency matrix rows and columns must share the same node order.")
    return adjacency_df


def _build_adjacency_map(adjacency_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    adjacency_map: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for target in adjacency_df.index:
        for source in adjacency_df.columns:
            if source == target:
                continue
            weight = float(adjacency_df.loc[target, source])
            if abs(weight) <= 1e-12:
                continue
            adjacency_map[source].append((target, weight))
    return adjacency_map


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _enumerate_paths(
    adjacency_map: dict[str, list[tuple[str, float]]],
    source_nodes: list[str],
    target_node: str,
    max_path_length: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def dfs(
        source_root: str,
        current_node: str,
        current_path: list[str],
        current_effect: float,
        visited: set[str],
    ) -> None:
        path_length = len(current_path) - 1
        if path_length > max_path_length:
            return
        if current_node == target_node:
            records.append(
                {
                    "source_root": source_root,
                    "path_nodes": list(current_path),
                    "path_length": path_length,
                    "path_effect": current_effect,
                }
            )
            return

        for next_node, edge_weight in adjacency_map.get(current_node, []):
            if next_node in visited:
                continue
            visited.add(next_node)
            current_path.append(next_node)
            dfs(
                source_root=source_root,
                current_node=next_node,
                current_path=current_path,
                current_effect=current_effect * edge_weight,
                visited=visited,
            )
            current_path.pop()
            visited.remove(next_node)

    for source in source_nodes:
        dfs(
            source_root=source,
            current_node=source,
            current_path=[source],
            current_effect=1.0,
            visited={source},
        )

    return records


def build_current_path_summary(
    adjacency_csv: str | Path,
    total_effect_csv: str | Path,
    max_path_length: int = 12,
    main_share_threshold: float = 0.8,
    max_main_paths_per_source: int = 3,
    min_main_path_effect_abs: float = 1e-3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    adjacency_df = _load_adjacency(adjacency_csv)
    totals_df = pd.read_csv(total_effect_csv)
    dictionary = build_variable_dictionary().set_index("node")
    source_nodes = [node for node in adjacency_df.columns if node != TARGET_NODE]
    total_effect_lookup = totals_df.set_index("node")["total_effect"].to_dict()

    raw_paths = _enumerate_paths(
        adjacency_map=_build_adjacency_map(adjacency_df),
        source_nodes=source_nodes,
        target_node=TARGET_NODE,
        max_path_length=max_path_length,
    )
    if not raw_paths:
        raise ValueError("No directed path from any source node to RISK was found in the adjacency matrix.")

    rows = []
    for idx, record in enumerate(raw_paths, start=1):
        path_nodes = record["path_nodes"]
        mediators = path_nodes[1:-1]
        mediator_groups = _ordered_unique([dictionary.loc[node, "group_lv1"] for node in mediators])
        rows.append(
            {
                "path_id": f"current_path_{idx:03d}",
                "path_str": "->".join(path_nodes),
                "source_root": record["source_root"],
                "mediator": "|".join(mediators) if mediators else "DIRECT",
                "target": TARGET_NODE,
                "path_length": record["path_length"],
                "path_sign": (
                    "positive"
                    if record["path_effect"] > 0
                    else ("negative" if record["path_effect"] < 0 else "zero")
                ),
                "path_effect": float(record["path_effect"]),
                "path_effect_abs": abs(float(record["path_effect"])),
                "source_group": dictionary.loc[record["source_root"], "group_lv1"],
                "mediator_group": "|".join(mediator_groups) if mediator_groups else "Direct",
                "target_group": dictionary.loc[TARGET_NODE, "group_lv1"],
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        by=["source_root", "path_effect_abs", "path_length", "path_str"],
        ascending=[True, False, True, True],
    )
    summary_df["source_total_effect"] = summary_df["source_root"].map(total_effect_lookup).astype(float)
    summary_df["source_abs_path_sum"] = summary_df.groupby("source_root")["path_effect_abs"].transform("sum")
    summary_df["path_rank_within_source"] = (
        summary_df.groupby("source_root")["path_effect_abs"].rank(method="first", ascending=False).astype(int)
    )
    summary_df["path_share_within_source_abs"] = summary_df["path_effect_abs"] / summary_df["source_abs_path_sum"]
    summary_df["is_main_path"] = False

    for source_root, group in summary_df.groupby("source_root", sort=False):
        group = group.sort_values(
            by=["path_effect_abs", "path_length", "path_str"],
            ascending=[False, True, True],
        )
        total_abs = float(group["path_effect_abs"].sum())
        cum_abs = 0.0
        n_selected = 0
        for row in group.itertuples(index=True):
            keep = False
            if n_selected < 1:
                keep = True
            elif (
                n_selected < max_main_paths_per_source
                and total_abs > 0.0
                and (cum_abs / total_abs) < main_share_threshold
                and row.path_effect_abs >= min_main_path_effect_abs
            ):
                keep = True

            if keep:
                summary_df.loc[row.Index, "is_main_path"] = True
                cum_abs += float(row.path_effect_abs)
                n_selected += 1

    summary_df = summary_df.sort_values(
        by=["path_effect_abs", "path_length", "source_root", "path_str"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    source_check = (
        summary_df.groupby("source_root", as_index=False)
        .agg(
            path_effect_sum=("path_effect", "sum"),
            path_effect_abs_sum=("path_effect_abs", "sum"),
            source_total_effect=("source_total_effect", "first"),
            n_paths=("path_id", "count"),
            n_main_paths=("is_main_path", "sum"),
        )
        .sort_values("source_root")
    )
    source_check["total_effect_delta"] = source_check["path_effect_sum"] - source_check["source_total_effect"]

    metadata = {
        "adjacency_csv": str(Path(adjacency_csv).resolve()),
        "total_effect_csv": str(Path(total_effect_csv).resolve()),
        "target_node": TARGET_NODE,
        "n_paths": int(len(summary_df)),
        "n_main_paths": int(summary_df["is_main_path"].sum()),
        "max_path_length": int(max_path_length),
        "main_path_rule": {
            "main_share_threshold": float(main_share_threshold),
            "max_main_paths_per_source": int(max_main_paths_per_source),
            "min_main_path_effect_abs": float(min_main_path_effect_abs),
        },
        "consistency_check": {
            "max_abs_total_effect_delta": float(source_check["total_effect_delta"].abs().max()),
            "per_source": source_check.to_dict(orient="records"),
        },
    }
    return summary_df, metadata


def write_current_path_summary(
    adjacency_csv: str | Path,
    total_effect_csv: str | Path,
    output_dir: str | Path | None = None,
    max_path_length: int = 12,
    main_share_threshold: float = 0.8,
    max_main_paths_per_source: int = 3,
    min_main_path_effect_abs: float = 1e-3,
) -> dict[str, Path]:
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, metadata = build_current_path_summary(
        adjacency_csv=adjacency_csv,
        total_effect_csv=total_effect_csv,
        max_path_length=max_path_length,
        main_share_threshold=main_share_threshold,
        max_main_paths_per_source=max_main_paths_per_source,
        min_main_path_effect_abs=min_main_path_effect_abs,
    )

    summary_path = out_dir / "current_path_summary.csv"
    metadata_path = out_dir / "current_path_summary_metadata.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary_path, "metadata": metadata_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize current-stage LiNGAM causal paths to the RISK node.")
    parser.add_argument(
        "--adjacency-csv",
        type=Path,
        default=DEFAULT_ADJACENCY_CSV,
        help="Current adjacency matrix CSV generated by run_current_lingam.py",
    )
    parser.add_argument(
        "--total-effect-csv",
        type=Path,
        default=DEFAULT_TOTAL_EFFECT_CSV,
        help="Current total effect table generated by run_current_lingam.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for current path summary outputs.",
    )
    parser.add_argument(
        "--max-path-length",
        type=int,
        default=12,
        help="Maximum directed path length to enumerate from each source node to RISK.",
    )
    parser.add_argument(
        "--main-share-threshold",
        type=float,
        default=0.8,
        help="Cumulative absolute-effect share used to retain main paths within each source node.",
    )
    parser.add_argument(
        "--max-main-paths-per-source",
        type=int,
        default=3,
        help="Upper bound on how many main paths can be retained for each source node.",
    )
    parser.add_argument(
        "--min-main-path-effect-abs",
        type=float,
        default=1e-3,
        help="Minimum absolute path effect to keep adding main paths beyond the first one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_current_path_summary(
        adjacency_csv=args.adjacency_csv,
        total_effect_csv=args.total_effect_csv,
        output_dir=args.output_dir,
        max_path_length=args.max_path_length,
        main_share_threshold=args.main_share_threshold,
        max_main_paths_per_source=args.max_main_paths_per_source,
        min_main_path_effect_abs=args.min_main_path_effect_abs,
    )
    print(f"[OK] current_path_summary.csv -> {paths['summary']}")
    print(f"[OK] current_path_summary_metadata.json -> {paths['metadata']}")


if __name__ == "__main__":
    main()
