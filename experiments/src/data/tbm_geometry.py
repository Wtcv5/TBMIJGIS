"""TBM 参数化几何模型 — 表面节点采样.

论文方案:
- cutterhead: 圆盘
- front shield / middle shield / tail shield: 圆柱面
- 表面节点按固定径向、周向和轴向分辨率采样
- 每个节点保留位置、法向和部件类型
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TBMSurface:
    """TBM 参数化表面节点集合."""

    positions: np.ndarray      # (M, 3) — 节点坐标 (初始位置)
    normals: np.ndarray        # (M, 3) — 节点外法向
    components: np.ndarray     # (M,)  — 部件类型: 0=cutterhead, 1=front, 2=middle, 3=tail

    def __len__(self) -> int:
        return len(self.positions)


COMPONENT_NAMES = {0: "cutterhead", 1: "front_shield", 2: "middle_shield", 3: "tail_shield"}
NUM_COMPONENTS = 4


def _sample_disk(radius: float, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """采样圆盘表面 (cutterhead). 返回 (positions, normals)."""
    r_step = int(radius / resolution)
    points = []
    normals = []
    for ir in range(r_step + 1):
        r = ir * resolution
        n_theta = max(8, int(2 * np.pi * r / resolution)) if r > 0 else 1
        for it in range(n_theta):
            theta = it * 2 * np.pi / n_theta
            points.append([0.0, r * np.cos(theta), r * np.sin(theta)])
            normals.append([1.0, 0.0, 0.0])  # 刀盘法向 = 推进方向
    return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)


def _sample_cylinder(length: float, radius: float, x_start: float, resolution: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """采样圆柱面 (shield). 返回 (positions, normals)."""
    n_ax = max(2, int(length / resolution))
    n_circ = max(12, int(2 * np.pi * radius / resolution))
    points = []
    normals = []
    for ia in range(n_ax + 1):
        x = x_start + ia * length / n_ax
        for ic in range(n_circ):
            theta = ic * 2 * np.pi / n_circ
            points.append([x, radius * np.cos(theta), radius * np.sin(theta)])
            normals.append([0.0, np.cos(theta), np.sin(theta)])  # 径向向外
    return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)


def build_tbm_surface(cutterhead_radius: float = 4.0,
                      shield_radius: float = 3.95,
                      front_len: float = 3.0,
                      middle_len: float = 3.5,
                      tail_len: float = 3.5,
                      resolution: float = 0.2) -> TBMSurface:
    """构建 TBM 参数化表面.

    X 轴正方向 = 推进方向.
    坐标原点处为 cutterhead 所在位置, shield 向 X 轴负方向延伸.
    """
    # Cutterhead at x=0
    ch_pos, ch_norm = _sample_disk(cutterhead_radius, resolution)
    ch_comp = np.full(len(ch_pos), 0, dtype=np.int32)

    # Front shield: x ∈ [-front_len, 0]
    fs_pos, fs_norm = _sample_cylinder(front_len, shield_radius, -front_len, resolution)
    fs_comp = np.full(len(fs_pos), 1, dtype=np.int32)

    # Middle shield: x ∈ [-(front_len+middle_len), -front_len]
    ms_start = -(front_len + middle_len)
    ms_pos, ms_norm = _sample_cylinder(middle_len, shield_radius, ms_start, resolution)
    ms_comp = np.full(len(ms_pos), 2, dtype=np.int32)

    # Tail shield: x ∈ [-(front_len+middle_len+tail_len), -(front_len+middle_len)]
    ts_start = -(front_len + middle_len + tail_len)
    ts_pos, ts_norm = _sample_cylinder(tail_len, shield_radius, ts_start, resolution)
    ts_comp = np.full(len(ts_pos), 3, dtype=np.int32)

    positions = np.concatenate([ch_pos, fs_pos, ms_pos, ts_pos], axis=0)
    normals = np.concatenate([ch_norm, fs_norm, ms_norm, ts_norm], axis=0)
    components = np.concatenate([ch_comp, fs_comp, ms_comp, ts_comp], axis=0)

    return TBMSurface(positions=positions, normals=normals, components=components)


def advance_surface(surface: TBMSurface, step_length: float,
                    advance_dir: np.ndarray) -> TBMSurface:
    """沿推进方向平移 TBM 表面一步.

    p_j(t+1) = p_j(t) + Δx · d
    """
    new_positions = surface.positions + step_length * np.asarray(advance_dir, dtype=np.float32)
    return TBMSurface(positions=new_positions, normals=surface.normals.copy(),
                      components=surface.components.copy())
