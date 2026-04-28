from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lingam_pipeline_v1.pre_attribution.attribution_config import (
    GROUP_COLORS,
    ensure_common_output_dir,
)


VARIABLE_SPECS = (
    {
        "node": "DK",
        "node_full_name": "Distance to karst",
        "source_column": "Distance_to_karst",
        "example_current_column": "Distance_to_karst",
        "example_future_pattern": "Distance_to_karst",
        "group_lv1": "Hydrogeology",
        "group_lv2": "Static geology",
        "is_target": False,
        "is_static": True,
        "is_dynamic": False,
    },
    {
        "node": "DB",
        "node_full_name": "Depth to bedrock",
        "source_column": "Depth_to_Bedrock",
        "example_current_column": "Depth_to_Bedrock",
        "example_future_pattern": "Depth_to_Bedrock",
        "group_lv1": "Hydrogeology",
        "group_lv2": "Static geology",
        "is_target": False,
        "is_static": True,
        "is_dynamic": False,
    },
    {
        "node": "DF",
        "node_full_name": "Distance to fault",
        "source_column": "Distance_to_Fault_m",
        "example_current_column": "Distance_to_Fault_m",
        "example_future_pattern": "Distance_to_Fault_m",
        "group_lv1": "Hydrogeology",
        "group_lv2": "Static geology",
        "is_target": False,
        "is_static": True,
        "is_dynamic": False,
    },
    {
        "node": "HDS",
        "node_full_name": "HDS",
        "source_column": "HDS",
        "example_current_column": "HDS_hist_2000_2010_2020",
        "example_future_pattern": "HDS_{t_minus_20}_{t}",
        "group_lv1": "Hydrogeology",
        "group_lv2": "Static geology",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "PR",
        "node_full_name": "Precipitation",
        "source_column": "Precip",
        "example_current_column": "Precip_hist_2000_2010_2020",
        "example_future_pattern": "Precip_{t_minus_20}_{t}",
        "group_lv1": "Climate",
        "group_lv2": "Climate forcing",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "TAS",
        "node_full_name": "Air temperature",
        "source_column": "Tas",
        "example_current_column": "Tas_hist_2000_2010_2020",
        "example_future_pattern": "Tas_{t_minus_20}_{t}",
        "group_lv1": "Climate",
        "group_lv2": "Climate forcing",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "HUSS",
        "node_full_name": "Specific humidity",
        "source_column": "Huss",
        "example_current_column": "Huss_hist_2000_2010_2020",
        "example_future_pattern": "Huss_{t_minus_20}_{t}",
        "group_lv1": "Climate",
        "group_lv2": "Climate forcing",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "PT",
        "node_full_name": "Population total",
        "source_column": "PopTotal",
        "example_current_column": "PopTotal_hist_2000_2010_2020",
        "example_future_pattern": "PopTotal_{t_minus_20}_{t}",
        "group_lv1": "Anthropogenic",
        "group_lv2": "Human pressure",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "UF",
        "node_full_name": "Urban fraction",
        "source_column": "UrbanFrac",
        "example_current_column": "UrbanFrac_hist_2000_2010_2020",
        "example_future_pattern": "UrbanFrac_{t_minus_20}_{t}",
        "group_lv1": "Anthropogenic",
        "group_lv2": "Human pressure",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "IP",
        "node_full_name": "Impervious index",
        "source_column": "ImperviousIndex",
        "example_current_column": "ImperviousIndex_hist_2000_2010_2020",
        "example_future_pattern": "ImperviousIndex_{t_minus_20}_{t}",
        "group_lv1": "Anthropogenic",
        "group_lv2": "Human pressure",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "LAI",
        "node_full_name": "Leaf area index",
        "source_column": "LAI",
        "example_current_column": "LAI_hist_2000_2010_2020",
        "example_future_pattern": "LAI_{t_minus_20}_{t}",
        "group_lv1": "Anthropogenic",
        "group_lv2": "Environmental mediator",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "WTD",
        "node_full_name": "Water table depth",
        "source_column": "WTD",
        "example_current_column": "WTD_hist_2000_2010_2020",
        "example_future_pattern": "WTD_{t_minus_20}_{t}",
        "group_lv1": "Anthropogenic",
        "group_lv2": "Environmental mediator",
        "is_target": False,
        "is_static": False,
        "is_dynamic": True,
    },
    {
        "node": "RISK",
        "node_full_name": "Predicted sinkhole probability",
        "source_column": "risk_probability",
        "example_current_column": "risk_probability",
        "example_future_pattern": "risk_probability",
        "group_lv1": "Target",
        "group_lv2": "Target",
        "is_target": True,
        "is_static": False,
        "is_dynamic": False,
    },
)


def build_variable_dictionary() -> pd.DataFrame:
    records = []
    for idx, spec in enumerate(VARIABLE_SPECS, start=1):
        record = dict(spec)
        record["plot_order"] = idx
        record["node_order"] = idx
        record["color_hex"] = GROUP_COLORS[record["group_lv1"]]
        records.append(record)
    return pd.DataFrame.from_records(records)


def node_order() -> list[str]:
    return [spec["node"] for spec in VARIABLE_SPECS]


def write_variable_dictionary(output_dir: str | Path | None = None) -> dict[str, Path]:
    out_dir = ensure_common_output_dir(output_dir)
    df = build_variable_dictionary()
    csv_path = out_dir / "variable_dictionary.csv"
    json_path = out_dir / "variable_dictionary.json"
    node_path = out_dir / "variable_nodes.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    node_path.write_text(json.dumps(node_order(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "csv": csv_path,
        "json": json_path,
        "nodes": node_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the attribution variable dictionary.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for variable dictionary outputs. Defaults to outputs/common.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_variable_dictionary(args.output_dir)
    print(f"[OK] variable_dictionary.csv -> {paths['csv']}")
    print(f"[OK] variable_dictionary.json -> {paths['json']}")
    print(f"[OK] variable_nodes.json -> {paths['nodes']}")


if __name__ == "__main__":
    main()

