"""Collect explicit rock--TBM spatial interaction descriptors.

This script implements the revised main experiment evidence:

- A_c(t): component-level geometric exposure;
- I_c(t): geometry-weighted TSP anomaly intensity;
- e_{t+h}: persistence residuals of TBM monitoring responses;
- descriptor-residual association statistics.

It does not load or require a trained model checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.context import TARGET_NAMES, build_descriptor_context
from src.diagnostics.spatial_interaction import (
    COMPONENT_NAMES,
    component_descriptors_for_snapshot,
    high_overlap_rate,
    persistence_residuals,
    safe_pearson,
    safe_spearman,
)


def save_json(data: dict[str, Any], path: Path) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def descriptor_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    residuals = persistence_residuals(context["y_test"], context["X_test"])

    for sample_idx, (chainage, graph_seq, residual, target) in enumerate(
        zip(context["test_chainages"], context["test_graph_seqs"], residuals, context["y_test"])
    ):
        snapshot = graph_seq[-1]
        descriptors = component_descriptors_for_snapshot(
            snapshot,
            anomaly_reference=context.get("vp_anomaly_reference"),
        )
        rock_node_count = int(snapshot.rock_attrs.shape[0])
        tbm_node_count = int(snapshot.tbm_components.shape[0])
        for item in descriptors:
            # The snapshot chainage is the last observed input step. The
            # diagnostic chainage is the target response step t+h.
            row = {
                "sample_idx": sample_idx,
                "chainage": float(chainage),
                "descriptor_snapshot_chainage": item.chainage,
                "component": item.component,
                "component_id": item.component_id,
                "rock_node_count": rock_node_count,
                "tbm_node_count": tbm_node_count,
                "node_count": item.node_count,
                "candidate_edge_count": item.candidate_edge_count,
                "A_geometric_exposure": item.geometric_exposure,
                "I_interaction_intensity": item.interaction_intensity,
                "weighted_anomaly_sum": item.weighted_anomaly_sum,
                "mean_anomaly": item.mean_anomaly,
            }
            for name, value in zip(TARGET_NAMES, residual):
                row[f"residual_{name}"] = float(value)
            for name, value in zip(TARGET_NAMES, target):
                row[f"target_{name}"] = float(value)
            rows.append(row)
    return rows


def association_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for component in COMPONENT_NAMES.values():
        comp_rows = [row for row in rows if row["component"] == component]
        if not comp_rows:
            continue
        for descriptor_col in ["A_geometric_exposure", "I_interaction_intensity"]:
            descriptor_values = np.asarray([row[descriptor_col] for row in comp_rows], dtype=np.float64)
            for target_name in TARGET_NAMES:
                residual_values = np.asarray(
                    [row[f"residual_{target_name}"] for row in comp_rows],
                    dtype=np.float64,
                )
                pr, pp = safe_pearson(descriptor_values, residual_values)
                sr, sp = safe_spearman(descriptor_values, residual_values)
                overlap = high_overlap_rate(descriptor_values, residual_values)
                results.append(
                    {
                        "component": component,
                        "descriptor": descriptor_col,
                        "response": target_name,
                        "n": len(comp_rows),
                        "pearson_r": pr,
                        "pearson_p": pp,
                        "spearman_r": sr,
                        "spearman_p": sp,
                        "high_descriptor_high_abs_residual_overlap": overlap,
                    }
                )
    return results


def component_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for component in COMPONENT_NAMES.values():
        comp_rows = [row for row in rows if row["component"] == component]
        if not comp_rows:
            continue
        for col in ["candidate_edge_count", "A_geometric_exposure", "I_interaction_intensity"]:
            values = np.asarray([row[col] for row in comp_rows], dtype=np.float64)
            summary.append(
                {
                    "component": component,
                    "quantity": col,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return summary


def graph_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    sample_ids = sorted({row["sample_idx"] for row in rows})
    for sample_idx in sample_ids:
        sample_rows = [row for row in rows if row["sample_idx"] == sample_idx]
        if not sample_rows:
            continue
        total_edges = sum(row["candidate_edge_count"] for row in sample_rows)
        total_exposure = sum(row["A_geometric_exposure"] for row in sample_rows)
        cutter_edges = sum(row["candidate_edge_count"] for row in sample_rows if row["component"] == "cutterhead")
        shield_edges = total_edges - cutter_edges
        out = {
            "sample_idx": sample_idx,
            "chainage": sample_rows[0]["chainage"],
            "descriptor_snapshot_chainage": sample_rows[0]["descriptor_snapshot_chainage"],
            "rock_node_count": sample_rows[0]["rock_node_count"],
            "tbm_node_count": sample_rows[0]["tbm_node_count"],
            "total_candidate_edges": total_edges,
            "total_geometric_exposure": total_exposure,
            "cutterhead_edge_share": cutter_edges / total_edges if total_edges else 0.0,
            "shield_edge_share": shield_edges / total_edges if total_edges else 0.0,
        }
        for row in sample_rows:
            comp = row["component"]
            out[f"{comp}_edges"] = row["candidate_edge_count"]
            out[f"{comp}_edge_share"] = row["candidate_edge_count"] / total_edges if total_edges else 0.0
            out[f"{comp}_A"] = row["A_geometric_exposure"]
            out[f"{comp}_I"] = row["I_interaction_intensity"]
        summary.append(out)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect explicit spatial interaction descriptors.")
    parser.add_argument("--config", default="config/bsll_dyk1017_205.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(__file__).resolve().parent.parent
    config_path = exp_dir / args.config
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    case_id = cfg.get("case", {}).get("id", Path(args.config).stem)
    if args.output_dir is None:
        output_dir = exp_dir / "outputs" / "descriptors" / case_id
    else:
        output_dir = exp_dir / args.output_dir

    context = build_descriptor_context(cfg, args.device)
    rows = descriptor_rows(context)
    assoc = association_rows(rows)
    summary = component_summary(rows)
    graph_rows = graph_summary(rows)

    write_csv(rows, output_dir / "component_spatial_descriptors.csv")
    write_csv(assoc, output_dir / "descriptor_residual_association.csv")
    write_csv(summary, output_dir / "component_descriptor_summary.csv")
    write_csv(graph_rows, output_dir / "graph_construction_summary.csv")
    save_json(
        {
            "case": cfg.get("case", {}),
            "config": str(config_path.relative_to(exp_dir)),
            "split_counts": context["split_counts"],
            "descriptor_definition": {
                "A_c(t)": "sum of geometry weights w_ij over candidate relations incident to component c",
                "I_c(t)": "geometry-weighted mean q_i over candidate relations incident to component c",
                "w_ij": "exp(-distance/tau_edge) * kappa",
                "q_i": "[0, 1] low-velocity anomaly score using training active-zone Q5/Q95 Vp reference",
                "residual": "r_{t+h}^{(k)} - r_t^{(k)} using the last observed monitoring response",
            },
            "vp_anomaly_reference": context.get("vp_anomaly_reference", {}),
            "component_summary": summary,
            "graph_construction_summary": graph_rows,
            "association": assoc,
        },
        output_dir / "descriptor_summary.json",
    )

    print(f"Saved descriptors to {output_dir}")
    print(f"Rows: descriptors={len(rows)}, associations={len(assoc)}")


if __name__ == "__main__":
    main()
