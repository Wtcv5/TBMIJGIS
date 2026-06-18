"""图序列构建 — 实现论文 Algorithm 1.

Geometry-constrained rock-TBM graph sequence construction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.data.alignment import ExcavationStep
from src.data.tbm_geometry import TBMSurface, advance_surface
from src.graph import edges, nodes


@dataclass
class GraphSnapshot:
    """单步图快照 G_t."""
    step_idx: int
    chainage: float
    hetero_data: HeteroData
    rock_attrs: torch.Tensor      # 当前 active rock 属性 (用于边特征)
    tbm_components: torch.Tensor  # TBM 部件 onehot (用于边特征)


def _compute_shield_tail_x(current_surface: TBMSurface) -> float:
    """从当前 TBM 表面节点计算盾尾 x 坐标.

    盾尾 = components==3 (tail_shield) 中最小的 x 值.
    """
    tail_mask = current_surface.components == 3
    if tail_mask.any():
        return float(current_surface.positions[tail_mask, 0].min())
    # 如果没有 tail shield 节点，用 front+middle 的最小 x 近似
    shield_mask = current_surface.components >= 1
    if shield_mask.any():
        return float(current_surface.positions[shield_mask, 0].min())
    return float(current_surface.positions[:, 0].min())


def build_graph_snapshot(step: ExcavationStep,
                         rock_coords: np.ndarray,
                         rock_attrs: np.ndarray,
                         excavated_mask: np.ndarray,
                         tbm_surface: TBMSurface,
                         cutterhead_look_ahead: float,
                         cutterhead_radius: float,
                         shield_radius: float,
                         tau_zone: float,
                         tau_edge: float,
                         eta_min: float,
                         device: str = "cpu") -> GraphSnapshot:
    """构建单个掘进步的图快照 G_t.

    对应 Algorithm 1 的 for each step 循环体.

    Args:
      tau_zone: active zone 径向扩展范围 (m), 用于节点筛选.
      tau_edge: 岩-机候选边距离阈值 (m), 用于边构建.
                tau_zone > tau_edge, 解耦径向范围与交互距离.
    """
    # 1. 平移 TBM 到当前位置
    current_surface = advance_surface(tbm_surface, step.x_face[0], np.array([1, 0, 0]))

    # 计算当前步的盾尾 x 坐标
    shield_tail_x = _compute_shield_tail_x(current_surface)

    # 2. 筛选 active zone 内未开挖岩体 (分部件 active zone, 使用 tau_zone)
    active_idx = np.where(
        nodes.active_rock_mask(
            rock_coords, step.x_face, excavated_mask,
            cutterhead_look_ahead=cutterhead_look_ahead,
            cutterhead_radius=cutterhead_radius,
            shield_radius=shield_radius,
            shield_tail_x=shield_tail_x,
            tau_zone=tau_zone,
        )
    )[0]

    # 3-4. 构建节点特征
    x_rock, rock_node_ids = nodes.build_rock_nodes(rock_coords, rock_attrs,
                                                    active_idx, device)
    x_tbm, tbm_node_ids = nodes.build_tbm_nodes(current_surface, device)

    # 5-6. 构建同质边 (resolution_hint=None 时自动推断体素间距)
    rock_edge_index = torch.tensor(
        edges.build_rock_edges_26nn(rock_coords[active_idx], resolution_hint=None),
        dtype=torch.long, device=device
    )
    tbm_edge_index = torch.tensor(
        edges.build_tbm_surface_edges(current_surface.positions),
        dtype=torch.long, device=device
    )

    # 7-9. 构建岩-机候选边 (使用 tau_edge)
    rm_edge_np, rm_attrs = edges.compute_rm_candidate_edges(
        rock_coords[active_idx], current_surface.positions,
        current_surface.normals, tau=tau_edge, eta_min=eta_min
    )

    data = HeteroData()
    data["rock"].x = x_rock
    data["tbm"].x = x_tbm
    data["rock"].node_ids = rock_node_ids
    data["tbm"].node_ids = tbm_node_ids
    data["rock", "adj", "rock"].edge_index = rock_edge_index
    data["tbm", "adj", "tbm"].edge_index = tbm_edge_index

    if rm_edge_np.shape[1] > 0:
        data["rock", "interact", "tbm"].edge_index = torch.tensor(
            rm_edge_np, dtype=torch.long, device=device
        )
        data["rock", "interact", "tbm"].edge_attrs = rm_attrs

    # 保存当前 active 岩体属性和 TBM 部件信息，便于后续边特征编码
    rock_attrs_t = torch.tensor(rock_attrs[active_idx], dtype=torch.float32, device=device)
    tbm_comp_onehot = torch.nn.functional.one_hot(
        torch.tensor(current_surface.components, dtype=torch.long, device=device),
        num_classes=4
    ).float()

    return GraphSnapshot(step_idx=step.step_idx, chainage=float(step.chainage),
                         hetero_data=data, rock_attrs=rock_attrs_t,
                         tbm_components=tbm_comp_onehot)


def build_graph_sequence(steps: List[ExcavationStep],
                         rock_coords: np.ndarray,
                         rock_attrs: np.ndarray,
                         tbm_surface: TBMSurface,
                         cutterhead_look_ahead: float = 5.0,
                         cutterhead_radius: float = 4.0,
                         shield_radius: float = 3.95,
                         tau_zone: float = 5.0,
                         tau_edge: float = 2.0,
                         eta_min: float = 0.3,
                         device: str = "cpu",
                         verbose: bool = True) -> List[GraphSnapshot]:
    """构建完整图序列 {G_1, ..., G_T}.

    对应 Algorithm 1 完整流程.

    Args:
      tau_zone: active zone 径向扩展范围 (m), 默认 5.0m (约 1.25 倍洞径).
      tau_edge: 岩-机候选边距离阈值 (m), 默认 2.0m (近场交互距离).
    """
    n_rock = len(rock_coords)
    excavated_mask = np.ones(n_rock, dtype=bool)  # True = 未开挖

    snapshots = []
    iterator = steps
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(steps, desc="Building graph sequence")

    for step in iterator:
        snap = build_graph_snapshot(step, rock_coords, rock_attrs,
                                    excavated_mask, tbm_surface,
                                    cutterhead_look_ahead, cutterhead_radius,
                                    shield_radius, tau_zone, tau_edge,
                                    eta_min, device)
        snapshots.append(snap)

        # 标记当前 TBM 已开挖区域为不可用
        # 仅排除圆柱内部 (x < x_face 且 r < R_s), 保留隧道轮廓外的岩体
        tbm_x = step.x_face[0]
        r_rock = np.sqrt(rock_coords[:, 1] ** 2 + rock_coords[:, 2] ** 2)
        behind_face_inside = (rock_coords[:, 0] < tbm_x) & (r_rock < shield_radius)
        excavated_mask[behind_face_inside] = False

    return snapshots
