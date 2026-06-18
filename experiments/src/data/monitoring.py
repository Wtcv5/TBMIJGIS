"""TBM 运行监测数据加载与预处理.

论文定义的监测变量:
  u_t = [AdvanceRate, RPM, Torque, Thrust, Penetration, ShieldPressure]
  r_{t+h} = [AdvanceRate, Torque, Thrust, Penetration, ShieldPressure]
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# 监测变量列名
FEATURE_COLS = ["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]
TARGET_COLS = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]


def load_monitoring(path: str | Path) -> pd.DataFrame:
    """加载 TBM 监测数据.

    期望 CSV 列: chainage, AdvanceRate, RPM, Torque, Thrust, Penetration,
                ShieldPressure
    """
    df = pd.read_csv(path)
    if "chainage" not in df.columns:
        raise KeyError("监测数据缺少 chainage 列")
    return df


def aggregate_by_chainage(df: pd.DataFrame, step_length: float = 1.0
                          ) -> pd.DataFrame:
    """按里程聚合监测数据.

    论文方案: 按 0.05-1.0 m 步长聚合原始高频采样.
    """
    df = df.copy()
    # 按步长分桶，取每桶均值
    df["chainage_bin"] = (df["chainage"] / step_length).round() * step_length
    agg = df.groupby("chainage_bin")[FEATURE_COLS].mean().reset_index()
    agg = agg.rename(columns={"chainage_bin": "chainage"})
    return agg


def align_chainage(tsp_coords: np.ndarray, monitoring_df: pd.DataFrame
                   ) -> pd.DataFrame:
    """局部坐标转换: x_local(t) = ShieldMileage(t) - ShieldMileage_start.

    将监测 chainage 对齐到 TSP 局部坐标系.
    """
    df = monitoring_df.copy()
    df["x_local"] = df["chainage"] - df["chainage"].min()
    return df


def standardize_monitoring(df: pd.DataFrame,
                           fit_df: Optional[pd.DataFrame] = None
                           ) -> Tuple[pd.DataFrame, dict]:
    """标准化监测变量, 返回 (标准化后的DataFrame, 标准化参数)."""
    df = df.copy()
    fit_df = df if fit_df is None else fit_df
    stats = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            mu = fit_df[col].mean()
            sigma = fit_df[col].std()
            df[col] = (df[col] - mu) / (sigma + 1e-8)
            stats[col] = {"mean": mu, "std": sigma}
    return df, stats


def build_monitoring_sequences(df: pd.DataFrame, K: int = 5, h: int = 1
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """构建监督学习样本.

    返回:
      X: (N, K, F)  — 历史监测序列
      y: (N, T)     — 未来响应标签
    其中 F=6 (feature cols), T=5 (target cols)
    """
    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    targets = df[TARGET_COLS].to_numpy(dtype=np.float32)
    # targets 中 RPM 不在目标列表中，但论文 r_{t+h} 也不含 RPM
    # 只使用 FEATURE_COLS 中出现在 TARGET_COLS 的列

    X_list, y_list = [], []
    for i in range(len(df) - K - h + 1):
        X_list.append(features[i:i + K])
        y_list.append(targets[i + K + h - 1])
    if not X_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)
