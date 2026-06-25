"""Sensitivity checks for explicit spatial interaction descriptors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_spatial_descriptors import association_rows, descriptor_rows
from src.diagnostics.context import build_descriptor_context


FIXED_SENSITIVITY_PAIRS = {
    "bsll_dyk1017_205": ("front_shield", "I_interaction_intensity", "AdvanceRate"),
    "bsll_dyk1017_205_h3": ("front_shield", "I_interaction_intensity", "AdvanceRate"),
    "sjls_dyk1252_411": ("cutterhead", "I_interaction_intensity", "ShieldPressure"),
}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def summarize_associations(case_id: str, variant: str, tau_edge: float, eta_min: float, assoc: list[dict[str, Any]]) -> dict[str, Any]:
    component, descriptor, response = FIXED_SENSITIVITY_PAIRS.get(
        case_id,
        ("cutterhead", "I_interaction_intensity", "ShieldPressure"),
    )
    fixed = next(
        (
            row for row in assoc
            if row["component"] == component
            and row["descriptor"] == descriptor
            and row["response"] == response
        ),
        {},
    )
    return {
        "case_id": case_id,
        "variant": variant,
        "tau_edge": tau_edge,
        "eta_min": eta_min,
        "fixed_component": component,
        "fixed_descriptor": descriptor,
        "fixed_response": response,
        "fixed_spearman_r": fixed.get("spearman_r", 0.0),
        "fixed_spearman_p": fixed.get("spearman_p", 1.0),
        "fixed_pearson_r": fixed.get("pearson_r", 0.0),
        "fixed_pearson_p": fixed.get("pearson_p", 1.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run descriptor sensitivity over graph thresholds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tau-edge-values", default="1.5,2.0,2.5,3.0")
    parser.add_argument("--eta-min-values", default="0.2,0.3,0.4")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(__file__).resolve().parent.parent
    config_path = exp_dir / args.config
    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    case_id = base_cfg.get("case", {}).get("id", Path(args.config).stem)
    output_dir = exp_dir / (args.output_dir or f"outputs/descriptors/{case_id}/sensitivity")

    tau_values = parse_float_list(args.tau_edge_values)
    eta_values = parse_float_list(args.eta_min_values)
    rows = []
    all_assoc = []
    for tau_edge in tau_values:
        for eta_min in eta_values:
            cfg = deepcopy(base_cfg)
            cfg["graph"]["tau_edge"] = tau_edge
            cfg["graph"]["normal_threshold"] = eta_min
            context = build_descriptor_context(cfg, args.device)
            desc_rows = descriptor_rows(context)
            assoc = association_rows(desc_rows)
            variant = f"tau{tau_edge:g}_eta{eta_min:g}"
            rows.append(summarize_associations(case_id, variant, tau_edge, eta_min, assoc))
            for row in assoc:
                all_assoc.append({
                    "case_id": case_id,
                    "variant": variant,
                    "tau_edge": tau_edge,
                    "eta_min": eta_min,
                    **row,
                })
            print(f"{case_id} {variant}: descriptors={len(desc_rows)} associations={len(assoc)}")

    write_csv(rows, output_dir / "descriptor_sensitivity_summary.csv")
    write_csv(all_assoc, output_dir / "descriptor_sensitivity_associations.csv")
    save_json({"case_id": case_id, "summary": rows}, output_dir / "descriptor_sensitivity_summary.json")
    print(f"Saved sensitivity outputs to {output_dir}")


if __name__ == "__main__":
    main()
