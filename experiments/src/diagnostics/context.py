"""Reusable context builder for explicit descriptor experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from src.data.alignment import build_excavation_steps, mileage_split
from src.data.monitoring import aggregate_by_chainage, load_monitoring, standardize_monitoring
from src.data.tbm_geometry import build_tbm_surface
from src.data.tsp_loader import TSP_ATTR_COLS, build_voxel_field, load_tsp, normalize_coords
from src.graph.sequence import build_graph_sequence
from src.training.graph_data import GraphSequenceDataset, collate_graph_sequence_batch


TARGET_NAMES = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]


def build_sequence_samples(graph_snapshots, input_values, target_values, K, h):
    X_mon, y_list, graph_samples, target_chainages = [], [], [], []
    for i in range(len(graph_snapshots) - K - h + 1):
        graph_samples.append(graph_snapshots[i:i + K])
        X_mon.append(input_values[i:i + K])
        y_list.append(target_values[i + K + h - 1])
        target_chainages.append(graph_snapshots[i + K + h - 1].chainage)
    if not X_mon:
        return [], np.array([]), np.array([]), np.array([])
    return (
        graph_samples,
        np.stack(X_mon),
        np.stack(y_list),
        np.asarray(target_chainages, dtype=np.float32),
    )


def monitoring_fit_rows_for_samples(sample_indices, n_rows: int, K: int, h: int) -> np.ndarray:
    rows = set()
    for idx in np.asarray(sample_indices, dtype=int):
        rows.update(range(idx, idx + K))
        target_idx = idx + K + h - 1
        if target_idx < n_rows:
            rows.add(target_idx)
    return np.asarray(sorted(rows), dtype=int)


def tsp_fit_stats_for_samples(tsp_df, sample_chainages: np.ndarray, sample_indices, margin: float = 5.0):
    tsp_norm = normalize_coords(tsp_df)
    x_local = tsp_norm["X_local"].to_numpy(dtype=np.float32)
    train_chainages = np.asarray(sample_chainages, dtype=np.float32)[np.asarray(sample_indices, dtype=int)]
    if train_chainages.size == 0:
        train_tsp_raw = tsp_norm[TSP_ATTR_COLS].to_numpy(dtype=np.float32)
    else:
        mask = np.zeros(len(tsp_norm), dtype=bool)
        for chainage in train_chainages:
            mask |= (x_local >= chainage - margin) & (x_local <= chainage + margin)
        if not mask.any():
            mask[:] = True
        train_tsp_raw = tsp_norm.loc[mask, TSP_ATTR_COLS].to_numpy(dtype=np.float32)
    attr_mean = train_tsp_raw.mean(axis=0, keepdims=True)
    attr_std = train_tsp_raw.std(axis=0, keepdims=True) + 1e-8
    return attr_mean, attr_std


def build_descriptor_context(cfg: dict[str, Any], device: str):
    tsp_df = load_tsp(cfg["data"]["tsp_path"])
    mon_df = load_monitoring(cfg["data"]["monitoring_path"])
    mon_df = aggregate_by_chainage(mon_df, cfg["excavation"]["step_length"])

    K = cfg["model"]["history_window"]
    h = cfg["model"]["predict_horizon"]
    n_total = len(mon_df) - K - h + 1
    if n_total <= 0:
        raise ValueError(f"Not enough monitoring rows for K={K}, h={h}.")

    split_cfg = cfg.get("split", {})
    split_method = split_cfg.get("method", "mileage")
    if split_method != "mileage":
        raise ValueError("Only mileage-ordered splitting is supported.")

    train_slice, val_slice, test_slice = mileage_split(
        n_total,
        train_r=split_cfg.get("train_ratio", 0.70),
        val_r=split_cfg.get("val_ratio", 0.15),
    )
    train_idx = np.arange(train_slice.start, train_slice.stop)
    val_idx = np.arange(val_slice.start, val_slice.stop)
    test_idx = np.arange(test_slice.start, test_slice.stop)
    sample_chainages_pre = np.asarray(
        [float(mon_df.iloc[i + K + h - 1]["chainage"]) for i in range(n_total)],
        dtype=np.float32,
    )

    train_monitor_rows = monitoring_fit_rows_for_samples(train_idx, len(mon_df), K, h)
    margin = max(
        float(cfg["graph"].get("tau_zone", 5.0)),
        float(cfg["excavation"].get("cutterhead_look_ahead", 5.0)),
    )
    attr_mean, attr_std = tsp_fit_stats_for_samples(tsp_df, sample_chainages_pre, train_idx, margin=margin)
    rock_coords, rock_attrs = build_voxel_field(tsp_df, attr_mean=attr_mean, attr_std=attr_std)
    mon_df, _ = standardize_monitoring(mon_df, fit_df=mon_df.iloc[train_monitor_rows].copy())

    tbm_cfg = cfg["tbm_geometry"]
    tbm_surface = build_tbm_surface(
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        front_len=tbm_cfg["front_shield_len"],
        middle_len=tbm_cfg["middle_shield_len"],
        tail_len=tbm_cfg["tail_shield_len"],
        resolution=tbm_cfg["surface_resolution"],
    )

    graph_cfg = cfg["graph"]
    tau_zone = graph_cfg.get("tau_zone", 5.0)
    tau_edge = graph_cfg.get("tau_edge", graph_cfg.get("distance_threshold", 2.0))
    eta_min = graph_cfg["normal_threshold"]
    steps = build_excavation_steps(mon_df, step_length=cfg["excavation"]["step_length"], K=K, h=h)
    snapshots = build_graph_sequence(
        steps,
        rock_coords,
        rock_attrs,
        tbm_surface,
        cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
        shield_radius=tbm_cfg["shield_radius"],
        tau_zone=tau_zone,
        tau_edge=tau_edge,
        eta_min=eta_min,
        device=device,
        verbose=False,
    )

    mon_features = mon_df[["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]].to_numpy(dtype=np.float32)
    mon_targets = mon_df[TARGET_NAMES].to_numpy(dtype=np.float32)
    graph_seqs, X_mon, y, sample_chainages = build_sequence_samples(snapshots, mon_features, mon_targets, K, h)

    test_gds = GraphSequenceDataset(
        [graph_seqs[i] for i in test_idx],
        X_mon[test_idx],
        y[test_idx],
        sample_chainages[test_idx],
    )
    test_loader = DataLoader(
        test_gds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_graph_sequence_batch,
    )

    return {
        "tsp_df": tsp_df,
        "test_loader": test_loader,
        "test_graph_seqs": [graph_seqs[i] for i in test_idx],
        "X_test": X_mon[test_idx],
        "y_test": y[test_idx],
        "test_chainages": sample_chainages[test_idx],
        "tau_edge": tau_edge,
        "rock_in_dim": snapshots[0].hetero_data["rock"].x.shape[1],
        "tbm_in_dim": snapshots[0].hetero_data["tbm"].x.shape[1],
        "edge_in_dim": 1 + 1 + 3 + rock_attrs.shape[1] + 4,
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "effective_samples": int(n_total),
        },
        "preprocessing_audit": {
            "split_method": split_method,
            "train_sample_indices": train_idx.tolist(),
            "val_sample_indices": val_idx.tolist(),
            "test_sample_indices": test_idx.tolist(),
            "train_target_chainages": sample_chainages_pre[train_idx].tolist(),
            "val_target_chainages": sample_chainages_pre[val_idx].tolist(),
            "test_target_chainages": sample_chainages_pre[test_idx].tolist(),
            "monitoring_scaler_fit_rows": train_monitor_rows.tolist(),
            "monitoring_scaler_fit_policy": "training sample historical input rows plus training target rows",
            "tsp_scaler_fit_policy": "TSP voxels within the active-zone margin of training target chainages",
            "tsp_scaler_margin_m": margin,
            "tsp_attr_mean": attr_mean.tolist(),
            "tsp_attr_std": attr_std.tolist(),
        },
    }

