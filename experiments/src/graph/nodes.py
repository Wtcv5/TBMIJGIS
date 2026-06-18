"""岩体节点与 TBM 表面节点构建.

论文定义:
  岩体节点: x_i^r(t) = [c_i, g_i, s_i(t)]
    c_i: 坐标, g_i: 地质属性, s_i(t): 开挖状态 (1=未开挖, 0=已开挖)
  TBM 表面节点: x_j^m(t) = [p_j(t), n_j(t), rho_j]
    p_j(t): 位置, n_j(t): 法向, rho_j: 部件类型

Active zone 定义 (分部件, 径向范围与候选边阈值解耦):
  刀盘: 0 ≤ x_i - x_face ≤ L_f  且 r_i ≤ R_c + τ_zone  (掌子面前方, 含径向约束)
  护盾: x_tail ≤ x_i ≤ x_face 且 R_s ≤ r_i ≤ R_s + τ_zone  (盾体周边环形)

  τ_zone: active zone 径向扩展范围 (物理含义: 应力重分布影响范围)
  τ_edge: 岩-机候选边距离阈值 (物理含义: 近场交互距离, 在 edges.py 中使用)
  两者解耦, τ_zone 通常 > τ_edge
"""

from typing import Tuple

import numpy as np
import torch

from src.data.tbm_geometry import TBMSurface


def active_rock_mask(rock_coords: np.ndarray, x_face: np.ndarray,
                     excavated_mask: np.ndarray,
                     cutterhead_look_ahead: float,
                     cutterhead_radius: float,
                     shield_radius: float, shield_tail_x: float,
                     tau_zone: float) -> np.ndarray:
    """筛选 active zone 内未开挖岩体节点.

    V_t^r = { i | c_i ∈ Ω_t, s_i(t) = 1 }

    Active zone 分部件定义:
      刀盘: 0 ≤ x_i - x_face[0] ≤ L_f 且 r_i ≤ R_c + τ_zone
      护盾: shield_tail_x ≤ x_i ≤ x_face[0] 且 R_s ≤ r_i ≤ R_s + τ_zone

    Args:
      tau_zone: active zone 径向扩展范围 (m), 与候选边距离阈值 tau_edge 解耦.
                通常设为 3-5m (反映应力重分布影响范围), 大于 tau_edge (2m).
    """
    x_face_val = x_face[0]
    x_rock = rock_coords[:, 0]
    r_rock = np.sqrt(rock_coords[:, 1] ** 2 + rock_coords[:, 2] ** 2)

    # 刀盘 active zone: 掌子面前方 [x_face, x_face + L_f], 径向 r ≤ R_c + τ_zone
    cutterhead_zone = (
        (x_rock >= x_face_val)
        & (x_rock <= x_face_val + cutterhead_look_ahead)
        & (r_rock <= cutterhead_radius + tau_zone)
    )

    # 护盾 active zone: 盾体周边环形 [shield_tail_x, x_face] × [R_s, R_s + τ_zone]
    shield_zone = (
        (x_rock >= shield_tail_x)
        & (x_rock <= x_face_val)
        & (r_rock >= shield_radius)
        & (r_rock <= shield_radius + tau_zone)
    )

    in_zone = cutterhead_zone | shield_zone
    return in_zone & excavated_mask


def build_rock_nodes(rock_coords: np.ndarray, rock_attrs: np.ndarray,
                     active_idx: np.ndarray, device: str = "cpu"
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建当前步的岩体节点特征.

    返回:
      x_rock: (N_active, 3 + D + 1) — [c_i, g_i, s_i(t)=1]
      node_ids: (N_active,) — 原始体素索引
    """
    coords = torch.tensor(rock_coords[active_idx], dtype=torch.float32, device=device)
    attrs = torch.tensor(rock_attrs[active_idx], dtype=torch.float32, device=device)
    # s_i(t) = 1 (未开挖)
    state = torch.ones(len(coords), 1, dtype=torch.float32, device=device)
    x_rock = torch.cat([coords, attrs, state], dim=1)
    node_ids = torch.tensor(active_idx, dtype=torch.long, device=device)
    return x_rock, node_ids


def build_tbm_nodes(surface: TBMSurface, device: str = "cpu"
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建 TBM 表面节点特征.

    返回:
      x_tbm: (M, 3 + 3 + 4) — [p_j(t), n_j(t), onehot(rho_j)]
      node_ids: (M,) — 原始表面节点索引
    """
    positions = torch.tensor(surface.positions, dtype=torch.float32, device=device)
    normals = torch.tensor(surface.normals, dtype=torch.float32, device=device)
    # One-hot 部件类型: 4 类
    comp = torch.tensor(surface.components, dtype=torch.long, device=device)
    comp_onehot = torch.nn.functional.one_hot(comp, num_classes=4).float()
    x_tbm = torch.cat([positions, normals, comp_onehot], dim=1)
    node_ids = torch.arange(len(positions), dtype=torch.long, device=device)
    return x_tbm, node_ids
