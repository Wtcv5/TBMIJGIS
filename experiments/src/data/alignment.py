"""Align TSP, TBM monitoring, and machine geometry by chainage."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class ExcavationStep:
    """Data associated with one chainage-indexed excavation step."""

    step_idx: int
    chainage: float
    x_face: np.ndarray
    u_t: np.ndarray
    r_target: Optional[np.ndarray] = None


@dataclass
class AlignedDataset:
    """Container for aligned excavation steps and TSP voxel data."""

    steps: List[ExcavationStep]
    tsp_coords: np.ndarray
    tsp_attrs: np.ndarray
    cutterhead_look_ahead: float = 5.0


def build_excavation_steps(
    monitoring_df: pd.DataFrame,
    step_length: float = 1.0,
    K: int = 5,
    h: int = 1,
) -> List[ExcavationStep]:
    """Build chainage-ordered excavation steps from monitoring data."""
    del step_length, K  # Kept in the public signature for existing callers.
    df = monitoring_df.sort_values("chainage").reset_index(drop=True)

    steps = []
    for idx, row in df.iterrows():
        chainage = float(row["chainage"])
        x_face = np.array(
            [chainage - float(df["chainage"].min()), 0.0, 0.0],
            dtype=np.float32,
        )
        u_t = np.array(
            [
                row.get("AdvanceRate", 0.0),
                row.get("RPM", 0.0),
                row.get("Torque", 0.0),
                row.get("Thrust", 0.0),
                row.get("Penetration", 0.0),
                row.get("ShieldPressure", 0.0),
            ],
            dtype=np.float32,
        )
        steps.append(ExcavationStep(idx, chainage, x_face, u_t))

    if h > 0:
        target_cols = [
            "AdvanceRate",
            "Torque",
            "Thrust",
            "Penetration",
            "ShieldPressure",
        ]
        for i, step in enumerate(steps):
            target_idx = i + h
            if target_idx < len(steps):
                row = df.iloc[target_idx]
                step.r_target = np.array(
                    [row.get(col, 0.0) for col in target_cols],
                    dtype=np.float32,
                )

    return steps


def mileage_split(
    n_total: int,
    train_r: float = 0.70,
    val_r: float = 0.15,
) -> tuple[slice, slice, slice]:
    """Return chronological train, validation, and test sample slices."""
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n_total),
    )
