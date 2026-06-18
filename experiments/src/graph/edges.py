"""图边构建.

论文定义三类边:
  E_t^{rr} — 岩体内部 26 邻域边
  E_t^{mm} — TBM 表面结构邻接边
  E_t^{rm} — 岩-机候选交互边 (几何约束筛选)
"""

from typing import Tuple

import numpy as np
import torch
from scipy.spatial import KDTree


# ── E_rr: 岩体内部边 ──────────────────────────────────────────────

def build_rock_edges_26nn(coords: np.ndarray, resolution_hint: float = None
                          ) -> np.ndarray:
    """构建岩体 26 邻域边.

    使用距离阈值 sqrt(3)*resolution 模拟 26 邻域.
    如果 resolution_hint 为 None, 从数据自动推断体素间距.
    返回 (2, E) 边索引数组.
    """
    if len(coords) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    tree = KDTree(coords)
    if resolution_hint is None:
        # 从最近邻距离推断体素间距
        dists, _ = tree.query(coords, k=2)
        resolution_hint = float(np.median(dists[:, 1]))
    radius = np.sqrt(3) * resolution_hint * 1.1
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    # 转为双向边
    edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return edges.T  # (2, E)


# ── E_mm: TBM 表面边 ──────────────────────────────────────────────

def build_tbm_surface_edges(positions: np.ndarray, resolution: float = 0.2
                            ) -> np.ndarray:
    """构建 TBM 表面网格邻接边.

    刀盘: 径向 + 周向邻接
    护盾: 轴向 + 周向邻接
    使用距离阈值统一处理.
    """
    tree = KDTree(positions)
    radius = resolution * 1.6  # 略大于采样分辨率以连接邻域
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return edges.T  # (2, E)


# ── E_rm: 岩-机候选边 ──────────────────────────────────────────────

def compute_rm_candidate_edges(rock_coords: np.ndarray, tbm_positions: np.ndarray,
                               tbm_normals: np.ndarray, tau: float = 2.0,
                               eta_min: float = 0.3,
                               x_face: np.ndarray = None) -> Tuple[np.ndarray, dict]:
    """构建岩-机候选边.

    条件:
      d_ij(t) ≤ τ      — 距离阈值
      κ_ij(t) ≥ η_min  — 法向一致性阈值
      c_i ∈ Ω_t        — active zone (由调用方保证)
      s_i(t) = 1       — 未开挖 (由调用方保证)

    d_ij(t) = || c_i - p_j(t) ||
    κ_ij(t) = max(0, n_j^T (c_i - p_j) / (d_ij + ε))

    返回:
      edge_index: (2, E_rm) — [rock_idx, tbm_idx]
      edge_attrs: dict with keys: distance, kappa, delta_vec, geometry_prior
    """
    tree = KDTree(tbm_positions)
    # 只保留距离 ≤ τ 的候选对
    pairs_list = tree.query_ball_point(rock_coords, r=tau, workers=-1)

    rock_idx_list = []
    tbm_idx_list = []
    distances = []
    kappas = []
    delta_vecs = []

    epsilon = 1e-8

    for i, neighbors in enumerate(pairs_list):
        if not neighbors:
            continue
        for j in neighbors:
            delta = rock_coords[i] - tbm_positions[j]
            d = float(np.linalg.norm(delta))
            if d < epsilon:
                continue
            # 法向一致性
            kappa = float(max(0.0, np.dot(tbm_normals[j], delta) / d))
            if kappa < eta_min:
                continue

            rock_idx_list.append(i)
            tbm_idx_list.append(j)
            distances.append(d)
            kappas.append(kappa)
            delta_vecs.append(delta)

    if not rock_idx_list:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attrs = {
            "distance": np.array([], dtype=np.float32),
            "kappa": np.array([], dtype=np.float32),
            "delta_vec": np.zeros((0, 3), dtype=np.float32),
            "geometry_prior": np.array([], dtype=np.float32),
        }
        return edge_index, edge_attrs

    edge_index = np.array([rock_idx_list, tbm_idx_list], dtype=np.int64)
    distances = np.array(distances, dtype=np.float32)
    kappas = np.array(kappas, dtype=np.float32)
    delta_vecs = np.array(delta_vecs, dtype=np.float32)

    # 几何先验: π_ij = exp(-d_ij/τ) · κ_ij
    geometry_prior = np.exp(-distances / tau) * kappas

    edge_attrs = {
        "distance": distances,
        "kappa": kappas,
        "delta_vec": delta_vecs,
        "geometry_prior": geometry_prior,
    }
    return edge_index, edge_attrs


# ── 边特征编码 (论文 Section 9) ──────────────────────────────────

def encode_rm_edge_features(distances: torch.Tensor, kappas: torch.Tensor,
                            delta_vecs: torch.Tensor, tau: float,
                            rock_attrs: torch.Tensor, tbm_comp_onehot: torch.Tensor,
                            edge_index: torch.Tensor) -> torch.Tensor:
    """编码岩-机边特征.

    a_ij^{rm} = [
      d_ij / τ,          几何兼容性
      κ_ij,
      (c_i - p_j) / τ,
      g_i,               地质上下文
      onehot(ρ_j)        机器部件语义
    ]

    返回: (E, 1 + 1 + 3 + D + 4) 边特征张量
    """
    d_norm = (distances / tau).unsqueeze(1)
    k_norm = kappas.unsqueeze(1)
    delta_norm = delta_vecs / tau

    src = edge_index[0]
    dst = edge_index[1]
    rock_attr_per_edge = rock_attrs[src]
    tbm_comp_per_edge = tbm_comp_onehot[dst]

    return torch.cat([d_norm, k_norm, delta_norm, rock_attr_per_edge, tbm_comp_per_edge], dim=1)
