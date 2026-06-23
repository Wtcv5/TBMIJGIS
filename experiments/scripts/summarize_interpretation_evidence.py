"""Summarize post-hoc interpretation evidence across tunnel cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE = [
    "outputs/evidence/bsll_dyk1017_205/posthoc_evidence.json",
    "outputs/evidence/bsll_dyk1017_205_h3/posthoc_evidence.json",
    "outputs/evidence/sjls_dyk1252_411/posthoc_evidence.json",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_evidence(path: Path) -> dict[str, Any]:
    evidence = read_json(path)
    surface = evidence["surface_evidence"]
    full = surface["full_model"]
    geom = surface["geometry_only"]
    case_id = Path(evidence["run_dir"]).name

    strongest_response = ""
    strongest_abs = -1.0
    for name, stats in full.get("response_correlations", {}).items():
        val = abs(float(stats.get("pearson_r", 0.0)))
        if val > strongest_abs:
            strongest_abs = val
            strongest_response = name

    dominant_component = ""
    dominant_share = -1.0
    for name, stats in full.get("component_summary", {}).items():
        share = float(stats.get("mean_share_relevance", 0.0))
        if share > dominant_share:
            dominant_share = share
            dominant_component = name

    return {
        "case_id": case_id,
        "evidence_path": str(path),
        "test_samples": evidence.get("split_counts", {}).get("test", ""),
        "prediction_mae_from_checkpoint": evidence.get("prediction_metrics_from_checkpoint", {}).get("mae", ""),
        "moran_full": full.get("morans_i_mean", ""),
        "moran_geometry_only": geom.get("morans_i_mean", ""),
        "component_cv_full": full.get("component_cv_mean", ""),
        "component_cv_geometry_only": geom.get("component_cv_mean", ""),
        "degree_control_r": full.get("degree_control_pearson_r_mean", ""),
        "relevance_geology_pearson_r": full.get("attention_geology_pearson_r", ""),
        "strongest_response_corr": strongest_response,
        "strongest_response_abs_pearson": strongest_abs,
        "dominant_component": dominant_component,
        "dominant_component_share": dominant_share,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize interpretation evidence.")
    parser.add_argument("--evidence", nargs="*", default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", default="outputs/evidence/interpretation_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(__file__).resolve().parent.parent
    rows = []
    for rel in args.evidence:
        path = exp_dir / rel
        if path.exists():
            rows.append(flatten_evidence(path))

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
