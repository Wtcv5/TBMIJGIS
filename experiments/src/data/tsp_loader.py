"""TSP 地质体素数据加载与预处理."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# 论文定义的 TSP 属性列
TSP_COLUMNS = ["X", "Y", "Z", "Vp", "Vs", "E", "Vp_Vs", "Pr", "rho"]
TSP_COORD_COLS = ["X", "Y", "Z"]
TSP_ATTR_COLS = ["Vp", "Vs", "E", "Vp_Vs", "Pr", "rho"]


def load_tsp(path: str | Path) -> pd.DataFrame:
    """加载 TSP 体素数据.

    支持列名: X, Y, Z, Vp, Vs, E, Vp_Vs, Pr, rho (或 ro, 自动映射)
    """
    df = pd.read_csv(path)
    # 兼容 ro → rho 列名
    if "ro" in df.columns and "rho" not in df.columns:
        df = df.rename(columns={"ro": "rho"})
    for col in TSP_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"TSP 数据缺少列: {col}")
    return df


def normalize_coords(df: pd.DataFrame) -> pd.DataFrame:
    """将 TSP 坐标归一化，X 轴与掘进方向对齐."""
    df = df.copy()
    df["X_local"] = df["X"] - df["X"].min()
    return df


def get_rock_coords(df: pd.DataFrame) -> np.ndarray:
    """返回 (N, 3) 岩体体素坐标数组."""
    return df[["X_local", "Y", "Z"]].to_numpy(dtype=np.float32)


def get_rock_attrs(df: pd.DataFrame,
                   mean: Optional[np.ndarray] = None,
                   std: Optional[np.ndarray] = None) -> np.ndarray:
    """返回 (N, D) 岩体属性数组，已标准化.

    属性: Vp, Vs, E, Vp_Vs, Pr, rho

    Args:
      mean: (1, D) 均值数组. 若为 None 则从当前 df 计算 (仅训练集应如此).
      std:  (1, D) 标准差数组. 若为 None 则从当前 df 计算.
    """
    raw = df[TSP_ATTR_COLS].to_numpy(dtype=np.float32)
    if mean is None:
        mean = raw.mean(axis=0, keepdims=True)
    if std is None:
        std = raw.std(axis=0, keepdims=True) + 1e-8
    return (raw - mean) / std


def build_voxel_field(df: pd.DataFrame,
                      attr_mean: Optional[np.ndarray] = None,
                      attr_std: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """构建体素场: 返回 (coords, attrs).

    coords: (N, 3) — 归一化后的坐标
    attrs: (N, 6)  — 标准化后的地质属性

    Args:
      attr_mean: (1, D) 训练集属性均值. 若为 None 则从全部数据计算.
      attr_std:  (1, D) 训练集属性标准差. 若为 None 则从全部数据计算.
    """
    df = normalize_coords(df)
    coords = get_rock_coords(df)
    attrs = get_rock_attrs(df, mean=attr_mean, std=attr_std)
    return coords, attrs
