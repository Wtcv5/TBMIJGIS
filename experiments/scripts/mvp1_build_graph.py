"""MVP 1: 图构建与可视化.

TSP CSV → TBM surface sampling → KDTree search → normal filtering
→ graph snapshot → visualization
"""

import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tsp_loader import load_tsp, build_voxel_field
from src.data.monitoring import load_monitoring, aggregate_by_chainage
from src.data.alignment import build_excavation_steps
from src.data.tbm_geometry import build_tbm_surface
from src.graph.sequence import build_graph_snapshot
from src.visualization.graph_viz import plot_graph_snapshot, plot_edge_statistics


def main(config_path: str = "config/default.yaml"):
    # 加载配置
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path("outputs/mvp1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("Loading data...")
    tsp_df = load_tsp(cfg["data"]["tsp_path"])
    rock_coords, rock_attrs = build_voxel_field(tsp_df)
    mon_df = load_monitoring(cfg["data"]["monitoring_path"])
    mon_df = aggregate_by_chainage(mon_df, cfg["excavation"]["step_length"])

    print(f"  Rock voxels: {len(rock_coords)}")
    print(f"  Monitoring records: {len(mon_df)}")

    # 2. 构建 TBM 参数化几何
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

    # 3. 构建掘进步
    steps = build_excavation_steps(
        mon_df, step_length=cfg["excavation"]["step_length"],
        K=cfg["model"]["history_window"], h=cfg["model"]["predict_horizon"],
    )
    print(f"  Excavation steps: {len(steps)}")

    # 4. 构建几个示例图快照
    graph_cfg = cfg["graph"]
    excavated_mask = np.ones(len(rock_coords), dtype=bool)
    snapshots = []

    for step in steps[:min(20, len(steps))]:
        snap = build_graph_snapshot(
            step, rock_coords, rock_attrs, excavated_mask, tbm_surface,
            cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
            shield_radius=tbm_cfg["shield_radius"],
            tau=graph_cfg["distance_threshold"],
            eta_min=graph_cfg["normal_threshold"],
        )
        snapshots.append(snap)

        # 标记已开挖 (掌子面已通过的岩体)
        tbm_x = step.x_face[0]
        behind = rock_coords[:, 0] < tbm_x
        excavated_mask[behind] = False

    # 5. 可视化
    print("Generating visualizations...")
    for i, snap in enumerate(snapshots[:5]):
        data = snap.hetero_data
        rock_c = data["rock"].x[:, :3].cpu().numpy()
        tbm_p = data["tbm"].x[:, :3].cpu().numpy()

        if ("rock", "interact", "tbm") in data.edge_types:
            ei = data["rock", "interact", "tbm"].edge_index.cpu().numpy()
        else:
            ei = np.zeros((2, 0), dtype=np.int64)

        plot_graph_snapshot(rock_c, tbm_p, ei,
                            tbm_components=snap.tbm_components.cpu().numpy(),
                            save_path=str(output_dir / f"graph_snapshot_{i:03d}.png"))

    plot_edge_statistics(snapshots,
                         save_path=str(output_dir / "edge_statistics.png"))

    print(f"Done. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
