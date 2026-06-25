"""Explicit spatial interaction descriptors for rock--TBM graph snapshots.

These descriptors are the main algorithmic objects for the revised paper
positioning. They do not require trained model checkpoints. Candidate rock--machine edges
come from the existing geometry-constrained graph construction, and their
geometry prior is used as the explicit interaction weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.graph.sequence import GraphSnapshot


COMPONENT_NAMES = {
    0: "cutterhead",
    1: "front_shield",
    2: "middle_shield",
    3: "tail_shield",
}


@dataclass
class DescriptorResult:
    """Component-chainage descriptor values for one graph snapshot."""

    chainage: float
    component: str
    component_id: int
    node_count: int
    candidate_edge_count: int
    geometric_exposure: float
    interaction_intensity: float
    weighted_anomaly_sum: float
    mean_anomaly: float


def vp_anomaly_from_standardized_attrs(rock_attrs: np.ndarray) -> np.ndarray:
    """Return the main TSP-derived anomaly score q_i = -z(Vp_i).

    The existing preprocessing standardizes TSP attributes with Vp in column 0.
    Lower relative Vp therefore corresponds to larger anomaly score.
    """
    if rock_attrs.ndim != 2 or rock_attrs.shape[1] < 1:
        return np.zeros(len(rock_attrs), dtype=np.float32)
    return -rock_attrs[:, 0].astype(np.float32)


def component_descriptors_for_snapshot(
    snapshot: GraphSnapshot,
    anomaly_scores: np.ndarray | None = None,
    component_names: dict[int, str] | None = None,
    eps: float = 1e-8,
) -> list[DescriptorResult]:
    """Compute A_c(t) and I_c(t) for one graph snapshot.

    A_c(t) is the sum of geometry weights around component c. I_c(t) is the
    geometry-weighted mean TSP anomaly around component c.
    """
    if component_names is None:
        component_names = COMPONENT_NAMES

    data = snapshot.hetero_data
    tbm_components = snapshot.tbm_components.argmax(dim=1).detach().cpu().numpy()
    rock_attrs = snapshot.rock_attrs.detach().cpu().numpy()
    if anomaly_scores is None:
        anomaly_scores = vp_anomaly_from_standardized_attrs(rock_attrs)
    else:
        anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)

    if ("rock", "interact", "tbm") not in data.edge_types:
        return [
            DescriptorResult(
                chainage=float(snapshot.chainage),
                component=name,
                component_id=int(component_id),
                node_count=int(np.sum(tbm_components == component_id)),
                candidate_edge_count=0,
                geometric_exposure=0.0,
                interaction_intensity=0.0,
                weighted_anomaly_sum=0.0,
                mean_anomaly=0.0,
            )
            for component_id, name in component_names.items()
        ]

    edge_store = data["rock", "interact", "tbm"]
    edge_index = edge_store.edge_index.detach().cpu().numpy()
    src = edge_index[0]
    dst = edge_index[1]
    weights = np.asarray(edge_store["edge_attrs"]["geometry_prior"], dtype=np.float32)
    if weights.ndim != 1:
        weights = weights.reshape(-1)

    rows: list[DescriptorResult] = []
    for component_id, name in component_names.items():
        node_mask = tbm_components == component_id
        edge_mask = node_mask[dst] if len(dst) else np.zeros(0, dtype=bool)
        w = weights[edge_mask]
        q = anomaly_scores[src[edge_mask]] if len(src) else np.zeros(0, dtype=np.float32)

        exposure = float(np.sum(w)) if len(w) else 0.0
        weighted_sum = float(np.sum(w * q)) if len(w) else 0.0
        intensity = weighted_sum / (exposure + eps) if exposure > 0 else 0.0
        mean_anomaly = float(np.mean(q)) if len(q) else 0.0

        rows.append(
            DescriptorResult(
                chainage=float(snapshot.chainage),
                component=name,
                component_id=int(component_id),
                node_count=int(np.sum(node_mask)),
                candidate_edge_count=int(np.sum(edge_mask)),
                geometric_exposure=exposure,
                interaction_intensity=float(intensity),
                weighted_anomaly_sum=weighted_sum,
                mean_anomaly=mean_anomaly,
            )
        )
    return rows


def descriptors_for_graph_sequences(
    graph_sequences: Iterable[list[GraphSnapshot]],
    component_names: dict[int, str] | None = None,
) -> list[DescriptorResult]:
    """Compute descriptors from the last snapshot of each sample sequence."""
    rows: list[DescriptorResult] = []
    for seq in graph_sequences:
        if not seq:
            continue
        rows.extend(component_descriptors_for_snapshot(seq[-1], component_names=component_names))
    return rows


def persistence_residuals(y: np.ndarray, monitoring_sequences: np.ndarray) -> np.ndarray:
    """Compute e_{t+h}^{(k)} = r_{t+h}^{(k)} - r_t^{(k)}.

    y columns are [AdvanceRate, Torque, Thrust, Penetration, ShieldPressure].
    monitoring columns are [AdvanceRate, RPM, Torque, Thrust, Penetration,
    ShieldPressure], so persistence uses columns [0, 2, 3, 4, 5] from the last
    observed monitoring row.
    """
    y = np.asarray(y, dtype=np.float32)
    mon = np.asarray(monitoring_sequences, dtype=np.float32)
    last_response = mon[:, -1, [0, 2, 3, 4, 5]]
    return y - last_response


def safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0, 1.0
    r, p = pearsonr(a, b)
    return float(r), float(p)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0, 1.0
    r, p = spearmanr(a, b)
    return float(r), float(p)


def high_overlap_rate(a: np.ndarray, b: np.ndarray, percentile: float = 75.0) -> float:
    """Overlap rate between high descriptor and high absolute residual intervals."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0:
        return 0.0
    a_high = a >= np.percentile(a, percentile)
    b_high = np.abs(b) >= np.percentile(np.abs(b), percentile)
    denom = int(np.sum(a_high))
    if denom == 0:
        return 0.0
    return float(np.sum(a_high & b_high) / denom)
