"""多源数据对齐 — 将 TSP 坐标、TBM 监测、TBM 几何统一到里程索引."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class ExcavationStep:
    """单个掘进步的数据结构.

    x_face(t)   当前掌子面/盾首位置
    G_t         当前岩-机交互图快照 (由 graph 模块填充)
    u_t         当前 TBM 运行监测向量
    r_{t+h}     未来 h 步 TBM 响应标签
    """
    step_idx: int
    chainage: float
    x_face: np.ndarray         # (3,) 当前盾首位置
    u_t: np.ndarray            # (F,) 标准化监测向量
    r_target: Optional[np.ndarray] = None  # (T,) 未来响应标签


@dataclass
class AlignedDataset:
    """对齐后的数据集."""
    steps: List[ExcavationStep]
    tsp_coords: np.ndarray     # (N, 3) 所有岩体体素坐标
    tsp_attrs: np.ndarray      # (N, D) 所有岩体属性
    cutterhead_look_ahead: float = 5.0


def build_excavation_steps(monitoring_df: pd.DataFrame,
                           step_length: float = 1.0,
                           K: int = 5, h: int = 1) -> List[ExcavationStep]:
    """从监测数据构建掘进步列表.

    按里程排序，每个 step 对应一个里程聚合步长.
    """
    df = monitoring_df.sort_values("chainage").reset_index(drop=True)

    steps = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        chainage = float(row["chainage"])
        x_face_local = chainage - df["chainage"].min()

        # TBM 盾首位置 (以 X 轴为推进方向)
        x_face = np.array([x_face_local, 0.0, 0.0], dtype=np.float32)

        # 当前监测向量 u_t
        u_t = np.array([
            row.get("AdvanceRate", 0.0),
            row.get("RPM", 0.0),
            row.get("Torque", 0.0),
            row.get("Thrust", 0.0),
            row.get("Penetration", 0.0),
            row.get("ShieldPressure", 0.0),
        ], dtype=np.float32)

        step = ExcavationStep(step_idx=idx, chainage=chainage,
                              x_face=x_face, u_t=u_t)
        steps.append(step)

    # 设置 r_target (未来 h 步标签)
    if h > 0:
        target_cols = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
        for i, step in enumerate(steps):
            target_idx = i + h
            if target_idx < len(steps):
                row = df.iloc[target_idx]
                step.r_target = np.array(
                    [row.get(c, 0.0) for c in target_cols], dtype=np.float32
                )

    return steps


def mileage_split(n_total: int, train_r: float = 0.70, val_r: float = 0.15
                  ) -> tuple:
    """按里程顺序划分训练/验证/测试索引.

    论文要求: 前 70% 训练, 中 15% 验证, 后 15% 测试. 不随机.
    """
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)
    # 剩余给 test
    return (slice(0, n_train), slice(n_train, n_train + n_val),
            slice(n_train + n_val, n_total))


def stratified_split_by_geology(monitoring_df: pd.DataFrame,
                                 tsp_df: pd.DataFrame,
                                 n_total: int,
                                 K: int = 5,
                                 h: int = 1,
                                 train_r: float = 0.70,
                                 val_r: float = 0.15,
                                 n_strata: int = 4,
                                 seed: int = 42) -> tuple:
    """基于地质特征的分层随机划分.

    每个样本 i 对应预测目标点 chainage = mon_df.iloc[i + K + h - 1].
    按该 chainage 处的 TSP Vs 分位数分层, 在每层内随机划分 70/15/15.

    这确保训练/验证/测试集的地质分布一致, 解决顺序划分导致的
    地质分布偏移问题 (测试区间 Vs shift=2.31 std).

    Args:
      monitoring_df: 监测数据 (含 chainage 列)
      tsp_df: TSP 地质数据 (含 X, Y, Z, Vs 等列)
      n_total: 样本数 (len(mon_df) - K - h + 1)
      K: 历史窗口长度
      h: 预测步长
      n_strata: 分层数 (默认 4: 低/中低/中高/高 Vs)
      seed: 随机种子

    Returns:
      (train_idx, val_idx, test_idx): 排序后的索引数组 (0 ~ n_total-1)
    """
    rng = np.random.RandomState(seed)

    # TSP 坐标归一化 (与 build_voxel_field 一致)
    tsp_norm = tsp_df.copy()
    tsp_norm["X_local"] = tsp_norm["X"] - tsp_norm["X"].min()

    # 为每个样本提取对应预测目标点的 TSP Vs 均值
    mon_sorted = monitoring_df.sort_values("chainage").reset_index(drop=True)
    vs_values = np.zeros(n_total, dtype=np.float32)

    for i in range(n_total):
        # 样本 i 的预测目标点
        target_idx = i + K + h - 1
        if target_idx < len(mon_sorted):
            chainage = mon_sorted.iloc[target_idx]["chainage"]
        else:
            chainage = mon_sorted.iloc[-1]["chainage"]
        # 取 chainage ±2m 范围内的 TSP 体素 Vs 均值
        mask = (tsp_norm["X_local"] >= chainage - 2.0) & (tsp_norm["X_local"] <= chainage + 2.0)
        if mask.sum() > 0:
            vs_values[i] = tsp_norm.loc[mask, "Vs"].mean()
        else:
            vs_values[i] = tsp_norm["Vs"].mean()

    # 按 Vs 分位数分层
    vs_sorted_idx = np.argsort(vs_values)
    strata_assignments = np.zeros(n_total, dtype=int)
    for s in range(n_strata):
        lo = int(s * n_total / n_strata)
        hi = int((s + 1) * n_total / n_strata)
        strata_assignments[vs_sorted_idx[lo:hi]] = s

    # 在每层内随机划分
    train_idx, val_idx, test_idx = [], [], []
    for s in range(n_strata):
        stratum_mask = strata_assignments == s
        stratum_indices = np.where(stratum_mask)[0]
        rng.shuffle(stratum_indices)

        n_s = len(stratum_indices)
        n_train_s = max(1, int(n_s * train_r))
        n_val_s = max(1, int(n_s * val_r))

        train_idx.extend(stratum_indices[:n_train_s])
        val_idx.extend(stratum_indices[n_train_s:n_train_s + n_val_s])
        test_idx.extend(stratum_indices[n_train_s + n_val_s:])

    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(test_idx))

    return train_idx, val_idx, test_idx
