"""Generate the IJGIS-style 4-panel graph construction figure.

Usage: python scripts/gen_graph_construction_fig.py
"""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tsp_loader import load_tsp, build_voxel_field
from src.data.monitoring import load_monitoring, aggregate_by_chainage
from src.data.alignment import build_excavation_steps
from src.data.tbm_geometry import build_tbm_surface
from src.graph.sequence import build_graph_snapshot
from src.visualization.graph_viz import plot_graph_construction_figure


def main(config_path: str = "config/bsll_dyk1017_205.yaml"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path("outputs/graph_construction")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    tsp_df = load_tsp(cfg["data"]["tsp_path"])
    rock_coords, rock_attrs = build_voxel_field(tsp_df)
    mon_df = load_monitoring(cfg["data"]["monitoring_path"])
    mon_df = aggregate_by_chainage(mon_df, cfg["excavation"]["step_length"])

    print(f"  Rock voxels: {len(rock_coords)}")
    print(f"  Monitoring records: {len(mon_df)}")

    # 2. Build TBM surface
    tbm_cfg = cfg["tbm_geometry"]
    tbm_surface = build_tbm_surface(
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        front_len=tbm_cfg["front_shield_len"],
        middle_len=tbm_cfg["middle_shield_len"],
        tail_len=tbm_cfg["tail_shield_len"],
        resolution=tbm_cfg["surface_resolution"],
    )
    print(f"  TBM surface nodes: {len(tbm_surface)}")

    # 3. Build excavation steps
    steps = build_excavation_steps(
        mon_df, step_length=cfg["excavation"]["step_length"],
        K=cfg["model"]["history_window"], h=cfg["model"]["predict_horizon"],
    )
    print(f"  Excavation steps: {len(steps)}")

    # 4. Build graph snapshots (use ~30 steps for panel d)
    graph_cfg = cfg["graph"]
    excavated_mask = np.ones(len(rock_coords), dtype=bool)
    snapshots = []

    n_steps = min(30, len(steps))
    for step in steps[:n_steps]:
        snap = build_graph_snapshot(
            step, rock_coords, rock_attrs, excavated_mask, tbm_surface,
            cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
            shield_radius=tbm_cfg["shield_radius"],
            tau=graph_cfg["distance_threshold"],
            eta_min=graph_cfg["normal_threshold"],
        )
        snapshots.append(snap)

        # Mark excavated
        tbm_x = step.x_face[0]
        behind = rock_coords[:, 0] < tbm_x
        excavated_mask[behind] = False

    print(f"  Built {len(snapshots)} graph snapshots")

    # 5. Generate 4-panel figure
    print("Generating 4-panel graph construction figure...")
    mid_idx = len(snapshots) // 2
    plot_graph_construction_figure(
        snapshot=snapshots[mid_idx],
        snapshots=snapshots,
        chainage_indices=[0, mid_idx, len(snapshots) - 1],
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        front_len=tbm_cfg["front_shield_len"],
        middle_len=tbm_cfg["middle_shield_len"],
        tail_len=tbm_cfg["tail_shield_len"],
        cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
        tau=graph_cfg["distance_threshold"],
        save_path=str(output_dir / "graph_construction_4panel.png"),
    )

    print(f"Done. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
