"""Regenerate checkpoint-derived spatial figures without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.collect_evidence import build_test_context, make_model
from scripts.run_graph_sequence_case import collect_graph_attention
from src.visualization.hotspot import (
    aggregate_attention_to_surface,
    plot_hotspot_vs_response,
    plot_shield_hotspot,
)


TARGET_NAMES = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bsll_dyk1017_205.yaml")
    parser.add_argument("--run-dir", default="outputs/bsll_dyk1017_205")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(__file__).resolve().parent.parent
    config_path = exp_dir / args.config
    run_dir = exp_dir / args.run_dir

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    context = build_test_context(cfg, args.device)
    model = make_model(cfg, context, args.device)
    model.load_state_dict(torch.load(run_dir / "best_graph_model.pt", map_location=args.device))

    scores, edge_indices, n_tbm_nodes = collect_graph_attention(
        model,
        context["test_loader"],
        args.device,
        context["tau_edge"],
    )
    relevance = []
    for score, edge_index, n_nodes in zip(scores, edge_indices, n_tbm_nodes):
        c_mean, _ = aggregate_attention_to_surface(score, edge_index, n_nodes)
        relevance.append(c_mean)

    first_snapshot = context["test_graph_seqs"][0][-1]
    tbm_pos = first_snapshot.hetero_data["tbm"].x[:, :3].cpu().numpy()
    tbm_comp = first_snapshot.tbm_components.argmax(dim=1).cpu().numpy()
    first_chainage = float(context["test_chainages"][0])
    plot_shield_hotspot(
        tbm_pos,
        tbm_comp,
        relevance[0],
        save_path=str(run_dir / f"hotspot_chainage_{first_chainage:.0f}.pdf"),
    )

    mean_relevance = np.asarray([float(c.mean()) for c in relevance], dtype=np.float32)
    plot_hotspot_vs_response(
        context["test_chainages"].tolist(),
        mean_relevance,
        context["y_test"],
        TARGET_NAMES,
        save_path=str(run_dir / "chainage_evolution.pdf"),
    )


if __name__ == "__main__":
    main()
