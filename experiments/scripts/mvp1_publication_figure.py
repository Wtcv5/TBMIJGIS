"""Generate a publication-style composite graph figure for one excavation step."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.alignment import build_excavation_steps
from src.data.monitoring import aggregate_by_chainage, load_monitoring
from src.data.tbm_geometry import build_tbm_surface
from src.data.tsp_loader import load_tsp, normalize_coords
from src.graph.sequence import build_graph_snapshot
from src.visualization.graph_viz import plot_publication_graph_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--step-idx", type=int, default=None)
    parser.add_argument("--output", default="outputs/mvp1/publication_graph_figure.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    tsp_df = load_tsp(cfg["data"]["tsp_path"])
    tsp_df = normalize_coords(tsp_df)
    rock_coords = tsp_df[["X_local", "Y", "Z"]].to_numpy(dtype=np.float32)
    rock_attrs = tsp_df[["Vp", "Vs", "E", "Vp_Vs", "Pr", "rho"]].to_numpy(dtype=np.float32)
    rock_attrs = (rock_attrs - rock_attrs.mean(axis=0, keepdims=True)) / (rock_attrs.std(axis=0, keepdims=True) + 1e-8)
    raw_vp = tsp_df["Vp"].to_numpy(dtype=np.float32)

    mon_df = load_monitoring(cfg["data"]["monitoring_path"])
    mon_df = aggregate_by_chainage(mon_df, cfg["excavation"]["step_length"])
    steps = build_excavation_steps(
        mon_df,
        step_length=cfg["excavation"]["step_length"],
        K=cfg["model"]["history_window"],
        h=cfg["model"]["predict_horizon"],
    )
    if not steps:
        raise RuntimeError("No excavation steps were constructed from monitoring data.")

    target_step_idx = args.step_idx if args.step_idx is not None else len(steps) // 2
    if not 0 <= target_step_idx < len(steps):
        raise IndexError(f"step_idx {target_step_idx} out of range for {len(steps)} steps")

    tbm_cfg = cfg["tbm_geometry"]
    tbm_surface = build_tbm_surface(
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        front_len=tbm_cfg["front_shield_len"],
        middle_len=tbm_cfg["middle_shield_len"],
        tail_len=tbm_cfg["tail_shield_len"],
        resolution=tbm_cfg["surface_resolution"],
    )

    excavated_mask = np.ones(len(rock_coords), dtype=bool)
    snapshot = None
    for step in steps[:target_step_idx + 1]:
        snapshot = build_graph_snapshot(
            step,
            rock_coords,
            rock_attrs,
            excavated_mask,
            tbm_surface,
            cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
            shield_radius=tbm_cfg["shield_radius"],
            tau=cfg["graph"]["distance_threshold"],
            eta_min=cfg["graph"]["normal_threshold"],
        )
        # 标记已开挖 (掌子面已通过的岩体)
        tbm_x = step.x_face[0]
        behind = rock_coords[:, 0] < tbm_x
        excavated_mask[behind] = False

    if snapshot is None:
        raise RuntimeError("Failed to build the requested graph snapshot.")

    rock_node_ids = snapshot.hetero_data["rock"].node_ids.cpu().numpy()
    rock_scalar = raw_vp[rock_node_ids]

    print(f"Rendering step {target_step_idx} at chainage {snapshot.chainage:.1f} m ...")
    plot_publication_graph_figure(
        snapshot,
        rock_scalar=rock_scalar,
        history_window=cfg["model"]["history_window"],
        predict_horizon=cfg["model"]["predict_horizon"],
        save_path=str(output_path),
    )
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
