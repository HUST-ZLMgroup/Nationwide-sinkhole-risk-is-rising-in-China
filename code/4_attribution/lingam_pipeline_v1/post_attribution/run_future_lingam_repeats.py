from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lingam import DirectLiNGAM
from sklearn.preprocessing import StandardScaler

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    N_REPEATS,
    OUTPUT_ROOT,
    SAMPLE_RATIO,
)
from lingam_pipeline_v1.pre_attribution.variable_schema import build_variable_dictionary, node_order


DEFAULT_PRIOR_MATRIX = OUTPUT_ROOT / "common" / "prior_knowledge_matrix.csv"
TARGET_NODE = "RISK"


def _load_prior_matrix(prior_matrix_csv: str | Path, nodes: list[str]) -> np.ndarray:
    prior_df = pd.read_csv(prior_matrix_csv, index_col=0)
    if prior_df.index.tolist() != nodes or prior_df.columns.tolist() != nodes:
        prior_df = prior_df.reindex(index=nodes, columns=nodes)
    if prior_df.isna().any().any():
        raise ValueError("prior_knowledge_matrix.csv has missing values after node reindexing.")
    return prior_df.to_numpy(dtype=int)


def _future_sample_csv(output_root: str | Path, ssp: str, year: str) -> Path:
    return Path(output_root) / "future" / ssp / year / "future_samples_for_lingam.csv"


def _fit_one_repeat(
    sample_df: pd.DataFrame,
    prior_knowledge: np.ndarray,
    dictionary: pd.DataFrame,
    nodes: list[str],
    scenario_id: str,
    ssp: str,
    year: str,
    repeat_id: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    X = sample_df.loc[:, nodes].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("Future LiNGAM input contains NaN values.")

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    model = DirectLiNGAM(random_state=repeat_id, prior_knowledge=prior_knowledge)
    model.fit(Xz)

    B = np.asarray(model.adjacency_matrix_, dtype=float)
    try:
        W = np.linalg.inv(np.eye(B.shape[0]) - B)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(np.eye(B.shape[0]) - B)
    risk_idx = nodes.index(TARGET_NODE)
    total_to_risk = W[risk_idx, :].copy()
    total_to_risk[risk_idx] = 0.0

    edge_rows: list[dict[str, object]] = []
    for target_idx, target in enumerate(nodes):
        for source_idx, source in enumerate(nodes):
            if source == target:
                continue
            weight = float(B[target_idx, source_idx])
            if np.isclose(weight, 0.0, atol=1e-12):
                continue
            prior_value = int(prior_knowledge[target_idx, source_idx])
            edge_rows.append(
                {
                    "scenario_id": scenario_id,
                    "ssp": ssp,
                    "year": year,
                    "repeat_id": repeat_id,
                    "source": source,
                    "target": target,
                    "source_group": dictionary.loc[source, "group_lv1"],
                    "target_group": dictionary.loc[target, "group_lv1"],
                    "edge_present": 1,
                    "edge_sign": "positive" if weight > 0 else "negative",
                    "edge_weight": weight,
                    "edge_weight_abs": abs(weight),
                    "target_is_risk": bool(target == TARGET_NODE),
                    "passes_prior_check": bool(prior_value != 0),
                }
            )

    total_rows: list[dict[str, object]] = []
    for node_idx, node in enumerate(nodes):
        if node == TARGET_NODE:
            continue
        effect = float(total_to_risk[node_idx])
        total_rows.append(
            {
                "scenario_id": scenario_id,
                "ssp": ssp,
                "year": year,
                "repeat_id": repeat_id,
                "node": node,
                "group_lv1": dictionary.loc[node, "group_lv1"],
                "group_lv2": dictionary.loc[node, "group_lv2"],
                "total_effect": effect,
                "total_effect_abs": abs(effect),
                "effect_sign": "positive" if effect > 0 else ("negative" if effect < 0 else "zero"),
            }
        )

    return edge_rows, total_rows


def run_future_lingam_repeats(
    ssp: str,
    year: str,
    future_csv: str | Path,
    prior_matrix_csv: str | Path,
    n_repeats: int = N_REPEATS,
    sample_ratio: float = SAMPLE_RATIO,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be within (0, 1], got {sample_ratio}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    dictionary = build_variable_dictionary().set_index("node")
    nodes = node_order()
    future_df = pd.read_csv(future_csv)
    missing = [col for col in nodes if col not in future_df.columns]
    if missing:
        raise KeyError(f"future_samples_for_lingam.csv is missing node columns: {missing}")

    scenario_id = f"{ssp}_{year}"
    prior_knowledge = _load_prior_matrix(prior_matrix_csv, nodes)
    sample_n = int(np.floor(len(future_df) * float(sample_ratio)))
    sample_n = max(sample_n, 2)
    if sample_n > len(future_df):
        raise ValueError("sample_n exceeded the number of available future samples.")

    all_edge_rows: list[dict[str, object]] = []
    all_total_rows: list[dict[str, object]] = []
    repeat_summaries: list[dict[str, object]] = []

    for repeat_id in range(1, n_repeats + 1):
        rng = np.random.default_rng(random_state + repeat_id)
        sampled_idx = rng.choice(len(future_df), size=sample_n, replace=False)
        sample_df = future_df.iloc[np.sort(sampled_idx)].reset_index(drop=True)
        edge_rows, total_rows = _fit_one_repeat(
            sample_df=sample_df,
            prior_knowledge=prior_knowledge,
            dictionary=dictionary,
            nodes=nodes,
            scenario_id=scenario_id,
            ssp=ssp,
            year=year,
            repeat_id=repeat_id,
        )
        all_edge_rows.extend(edge_rows)
        all_total_rows.extend(total_rows)
        repeat_summaries.append(
            {
                "repeat_id": repeat_id,
                "n_samples": int(len(sample_df)),
                "n_edges": int(len(edge_rows)),
            }
        )

    edge_df = pd.DataFrame(all_edge_rows).sort_values(
        by=["repeat_id", "target_is_risk", "edge_weight_abs"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    total_df = pd.DataFrame(all_total_rows).sort_values(
        by=["repeat_id", "total_effect_abs"],
        ascending=[True, False],
    ).reset_index(drop=True)

    metadata = {
        "scenario_id": scenario_id,
        "ssp": ssp,
        "year": year,
        "future_csv": str(Path(future_csv).resolve()),
        "prior_matrix_csv": str(Path(prior_matrix_csv).resolve()),
        "n_total_rows": int(len(future_df)),
        "n_repeats": int(n_repeats),
        "sample_ratio": float(sample_ratio),
        "sample_n": int(sample_n),
        "random_state": int(random_state),
        "repeat_summaries": repeat_summaries,
        "n_edge_records": int(len(edge_df)),
        "n_total_effect_records": int(len(total_df)),
    }
    return edge_df, total_df, metadata


def write_future_lingam_repeats(
    ssp: str,
    year: str,
    future_csv: str | Path,
    prior_matrix_csv: str | Path,
    output_dir: str | Path | None = None,
    n_repeats: int = N_REPEATS,
    sample_ratio: float = SAMPLE_RATIO,
    random_state: int = 0,
) -> dict[str, Path]:
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / "future" / ssp / year
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_df, total_df, metadata = run_future_lingam_repeats(
        ssp=ssp,
        year=year,
        future_csv=future_csv,
        prior_matrix_csv=prior_matrix_csv,
        n_repeats=n_repeats,
        sample_ratio=sample_ratio,
        random_state=random_state,
    )

    edge_path = out_dir / "edge_records_long.csv"
    total_path = out_dir / "total_effect_records_long.csv"
    meta_path = out_dir / "future_lingam_repeats_metadata.json"
    edge_df.to_csv(edge_path, index=False, encoding="utf-8-sig")
    total_df.to_csv(total_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"edges": edge_path, "totals": total_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated future-stage LiNGAM fits with fixed-ratio sampling without replacement.")
    parser.add_argument("--ssp", required=True, choices=list(node for node in ["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"]))
    parser.add_argument("--year", required=True, choices=list(node for node in ["2040", "2060", "2080", "2100"]))
    parser.add_argument(
        "--future-csv",
        type=Path,
        default=None,
        help="Scenario-specific future sample table. Defaults to outputs/future/{ssp}/{year}/future_samples_for_lingam.csv",
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
        default=None,
        help="Scenario-specific output directory. Defaults to outputs/future/{ssp}/{year}",
    )
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS, help="Number of LiNGAM repeats.")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=SAMPLE_RATIO,
        help="Sampling ratio applied without replacement for each repeat.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Base random seed. Each repeat uses random_state + repeat_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    future_csv = args.future_csv or _future_sample_csv(OUTPUT_ROOT, args.ssp, args.year)
    output_dir = args.output_dir or (OUTPUT_ROOT / "future" / args.ssp / args.year)
    paths = write_future_lingam_repeats(
        ssp=args.ssp,
        year=args.year,
        future_csv=future_csv,
        prior_matrix_csv=args.prior_matrix_csv,
        output_dir=output_dir,
        n_repeats=args.n_repeats,
        sample_ratio=args.sample_ratio,
        random_state=args.random_state,
    )
    print(f"[OK] edge_records_long.csv -> {paths['edges']}")
    print(f"[OK] total_effect_records_long.csv -> {paths['totals']}")
    print(f"[OK] future_lingam_repeats_metadata.json -> {paths['metadata']}")


if __name__ == "__main__":
    main()
