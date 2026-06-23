"""Summarize graph-sequence results across tunnel cases.

This script is intentionally read-only. It collects the standard JSON outputs
from completed case runs and writes one compact table for manuscript planning.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CASE_CONFIGS = [
    "config/bsll_dyk1017_205.yaml",
    "config/bsll_dyk1017_205_h3.yaml",
    "config/sjls_dyk1252_411.yaml",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_case(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    case = cfg.get("case", {})
    runtime = cfg.get("runtime", {})
    output_dir = config_path.parent.parent / runtime.get("output_dir", "")
    if not (output_dir / "metrics_global.json").exists() and runtime.get("legacy_output_dir"):
        output_dir = config_path.parent.parent / runtime["legacy_output_dir"]
    metrics = read_json(output_dir / "metrics_global.json")
    ablation = read_json(output_dir / "ablation_metrics.json")
    audit = read_json(output_dir / "preprocessing_audit.json")

    full = metrics.get("Full Model", {})
    persistence = metrics.get("Persistence", {})
    random_edges = ablation.get("Random Edges", {})
    no_prior = ablation.get("No Geometry Prior", {})
    test_chainages = audit.get("test_target_chainages", [])
    train_chainages = audit.get("train_target_chainages", [])

    return {
        "case_id": case.get("id", config_path.stem),
        "tunnel": case.get("tunnel", ""),
        "start_chainage": case.get("start_chainage", ""),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "train_samples": len(train_chainages),
        "test_samples": len(test_chainages),
        "test_chainage_min": min(test_chainages) if test_chainages else "",
        "test_chainage_max": max(test_chainages) if test_chainages else "",
        "full_mae": full.get("mae", ""),
        "full_rmse": full.get("rmse", ""),
        "full_r2": full.get("r2", ""),
        "full_pearson": full.get("pearson", ""),
        "persistence_mae": persistence.get("mae", ""),
        "persistence_rmse": persistence.get("rmse", ""),
        "random_edges_mae": random_edges.get("mae", ""),
        "no_prior_mae": no_prior.get("mae", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed tunnel case results.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CASE_CONFIGS,
        help="Case config paths relative to experiments/.",
    )
    parser.add_argument("--output", default="outputs/case_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(__file__).resolve().parent.parent
    rows = [load_case(exp_dir / cfg) for cfg in args.configs]

    output_path = exp_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Wrote {output_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
