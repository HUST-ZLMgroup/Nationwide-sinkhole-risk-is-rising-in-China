from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from lingam_pipeline_v1.pre_attribution.attribution_config import FREQ_THRESHOLD, OUTPUT_ROOT


DEFAULT_EDGE_FREQUENCY_CSV = OUTPUT_ROOT / "future" / "summary" / "edge_frequency_summary.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "future" / "summary"
TARGET_NODE = "RISK"


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _enumerate_paths(edge_df: pd.DataFrame, max_path_length: int) -> list[dict[str, object]]:
    adjacency: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in edge_df.itertuples(index=False):
        adjacency[row.source].append(
            {
                "target": row.target,
                "mean_weight": float(row.mean_weight),
                "freq_present": float(row.freq_present),
                "source_group": row.source_group,
                "target_group": row.target_group,
            }
        )

    records: list[dict[str, object]] = []

    def dfs(
        source_root: str,
        current_node: str,
        current_path: list[str],
        current_effect: float,
        current_freq: float,
        visited: set[str],
    ) -> None:
        path_length = len(current_path) - 1
        if path_length > max_path_length:
            return
        if current_node == TARGET_NODE:
            records.append(
                {
                    "source_root": source_root,
                    "path_nodes": list(current_path),
                    "path_length": path_length,
                    "mean_path_effect": current_effect,
                    "freq_path": current_freq,
                }
            )
            return

        for edge in adjacency.get(current_node, []):
            next_node = edge["target"]
            if next_node in visited:
                continue
            visited.add(next_node)
            current_path.append(next_node)
            dfs(
                source_root=source_root,
                current_node=next_node,
                current_path=current_path,
                current_effect=current_effect * edge["mean_weight"],
                current_freq=min(current_freq, edge["freq_present"]),
                visited=visited,
            )
            current_path.pop()
            visited.remove(next_node)

    source_nodes = sorted(set(edge_df["source"]) - {TARGET_NODE})
    for source_root in source_nodes:
        dfs(
            source_root=source_root,
            current_node=source_root,
            current_path=[source_root],
            current_effect=1.0,
            current_freq=1.0,
            visited={source_root},
        )
    return records


def _path_group(source_group: str, path_length: int) -> str:
    if source_group == "Hydrogeology":
        return "Hydrogeology_to_RISK"
    if source_group == "Climate":
        return "Climate_mediated_to_RISK" if path_length > 1 else "Climate_to_RISK"
    if source_group == "Anthropogenic":
        return "Anthropogenic_mediated_to_RISK" if path_length > 1 else "Anthropogenic_to_RISK"
    return f"{source_group}_to_RISK"


def build_stable_path_outputs(
    edge_frequency_csv: str | Path = DEFAULT_EDGE_FREQUENCY_CSV,
    freq_threshold: float = FREQ_THRESHOLD,
    max_path_length: int = 12,
    main_share_threshold: float = 0.8,
    max_main_paths_per_source: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq_df = pd.read_csv(edge_frequency_csv)
    stable_edges = freq_df.loc[freq_df["is_stable_edge"]].copy()
    if stable_edges.empty:
        raise ValueError("No stable edges were found in edge_frequency_summary.csv.")

    path_rows: list[dict[str, object]] = []
    for (scenario_id, ssp, year), scenario_edges in stable_edges.groupby(["scenario_id", "ssp", "year"], sort=False):
        raw_paths = _enumerate_paths(scenario_edges, max_path_length=max_path_length)
        for idx, record in enumerate(raw_paths, start=1):
            path_nodes = record["path_nodes"]
            mediators = path_nodes[1:-1]
            mediator_groups = _ordered_unique(
                scenario_edges.loc[scenario_edges["source"].isin(mediators), "source_group"].tolist()
                + scenario_edges.loc[scenario_edges["target"].isin(mediators), "target_group"].tolist()
            )
            source_group = scenario_edges.loc[scenario_edges["source"] == record["source_root"], "source_group"].iloc[0]
            path_rows.append(
                {
                    "scenario_id": scenario_id,
                    "ssp": ssp,
                    "year": year,
                    "path_id": f"{scenario_id}_stable_path_{idx:03d}",
                    "path_str": "->".join(path_nodes),
                    "path_length": record["path_length"],
                    "source_root": record["source_root"],
                    "mediator": "|".join(mediators) if mediators else "DIRECT",
                    "target": TARGET_NODE,
                    "source_group": source_group,
                    "mediator_group": "|".join(mediator_groups) if mediator_groups else "Direct",
                    "target_group": "Target",
                    "freq_path": float(record["freq_path"]),
                    "mean_path_effect": float(record["mean_path_effect"]),
                    "mean_path_effect_abs": abs(float(record["mean_path_effect"])),
                    "path_sign": (
                        "positive"
                        if record["mean_path_effect"] > 0
                        else ("negative" if record["mean_path_effect"] < 0 else "zero")
                    ),
                }
            )

    if not path_rows:
        raise ValueError("Stable edges exist, but no directed stable paths to RISK were found.")

    path_df = pd.DataFrame(path_rows)
    path_df["is_stable_path"] = path_df["freq_path"] >= float(freq_threshold)
    path_df["is_main_path"] = False

    for _, group in path_df.groupby(["scenario_id", "source_root"], sort=False):
        group = group.sort_values(
            by=["mean_path_effect_abs", "path_length", "path_str"],
            ascending=[False, True, True],
        )
        total_abs = float(group["mean_path_effect_abs"].sum())
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
            ):
                keep = True
            if keep:
                path_df.loc[row.Index, "is_main_path"] = True
                cum_abs += float(row.mean_path_effect_abs)
                n_selected += 1

    path_df["path_group"] = path_df.apply(
        lambda row: _path_group(row["source_group"], int(row["path_length"])),
        axis=1,
    )
    path_df = path_df.sort_values(
        by=["scenario_id", "mean_path_effect_abs", "path_length", "path_str"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    contribution_df = (
        path_df.groupby(["scenario_id", "ssp", "year", "path_group"], as_index=False)
        .agg(
            contribution_mean=("mean_path_effect", "sum"),
            contribution_mean_abs=("mean_path_effect_abs", "sum"),
            n_stable_paths=("path_id", "count"),
        )
        .sort_values(["scenario_id", "contribution_mean_abs"], ascending=[True, False])
        .reset_index(drop=True)
    )
    contribution_df["contribution_share"] = contribution_df.groupby("scenario_id")["contribution_mean_abs"].transform(
        lambda s: s / s.sum() if float(s.sum()) > 0 else 0.0
    )

    edge_counts = []
    for (scenario_id, path_group), group in path_df.groupby(["scenario_id", "path_group"], sort=False):
        edge_set = set()
        for path_str in group["path_str"]:
            nodes = str(path_str).split("->")
            edge_set.update(zip(nodes[:-1], nodes[1:]))
        edge_counts.append(
            {
                "scenario_id": scenario_id,
                "path_group": path_group,
                "n_stable_edges": len(edge_set),
            }
        )
    edge_count_df = pd.DataFrame(edge_counts)
    contribution_df = contribution_df.merge(edge_count_df, on=["scenario_id", "path_group"], how="left")
    contribution_df["n_stable_edges"] = contribution_df["n_stable_edges"].fillna(0).astype(int)
    return path_df, contribution_df


def write_stable_path_outputs(
    edge_frequency_csv: str | Path = DEFAULT_EDGE_FREQUENCY_CSV,
    output_dir: str | Path | None = None,
    freq_threshold: float = FREQ_THRESHOLD,
    max_path_length: int = 12,
    main_share_threshold: float = 0.8,
    max_main_paths_per_source: int = 3,
) -> dict[str, Path]:
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    path_df, contribution_df = build_stable_path_outputs(
        edge_frequency_csv=edge_frequency_csv,
        freq_threshold=freq_threshold,
        max_path_length=max_path_length,
        main_share_threshold=main_share_threshold,
        max_main_paths_per_source=max_main_paths_per_source,
    )
    path_path = out_dir / "stable_path_summary.csv"
    contribution_path = out_dir / "group_path_contribution_summary.csv"
    path_df.to_csv(path_path, index=False, encoding="utf-8-sig")
    contribution_df.to_csv(contribution_path, index=False, encoding="utf-8-sig")
    return {"paths": path_path, "contributions": contribution_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stable future LiNGAM paths and grouped contributions.")
    parser.add_argument(
        "--edge-frequency-csv",
        type=Path,
        default=DEFAULT_EDGE_FREQUENCY_CSV,
        help="Aggregated edge_frequency_summary.csv produced by summarize_edge_frequency.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for stable path summary outputs.",
    )
    parser.add_argument("--freq-threshold", type=float, default=FREQ_THRESHOLD)
    parser.add_argument("--max-path-length", type=int, default=12)
    parser.add_argument("--main-share-threshold", type=float, default=0.8)
    parser.add_argument("--max-main-paths-per-source", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_stable_path_outputs(
        edge_frequency_csv=args.edge_frequency_csv,
        output_dir=args.output_dir,
        freq_threshold=args.freq_threshold,
        max_path_length=args.max_path_length,
        main_share_threshold=args.main_share_threshold,
        max_main_paths_per_source=args.max_main_paths_per_source,
    )
    print(f"[OK] stable_path_summary.csv -> {paths['paths']}")
    print(f"[OK] group_path_contribution_summary.csv -> {paths['contributions']}")


if __name__ == "__main__":
    main()
