"""Collect post-hoc evidence from a completed graph-sequence run.

The script rebuilds the configured dataset, loads a trained Full Model
checkpoint, recomputes test predictions, extracts attention-derived surface
relevance, and writes traceable evidence files for manuscript tables.

It is designed to strengthen the evidence chain without retraining:

- prediction metrics recomputed from the checkpoint;
- attention-geology correlation;
- Moran's I and component-level CV for surface relevance;
- degree-control correlation between relevance and incident edge count;
- geometry-only baseline computed from fixed geometry priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_graph_sequence_case import (
    build_sequence_samples,
    collect_graph_attention,
    collect_graph_predictions,
    monitoring_fit_rows_for_samples,
    tsp_fit_stats_for_samples,
)
from src.data.alignment import build_excavation_steps, mileage_split
from src.data.monitoring import aggregate_by_chainage, load_monitoring, standardize_monitoring
from src.data.tbm_geometry import build_tbm_surface
from src.data.tsp_loader import build_voxel_field, load_tsp, normalize_coords
from src.graph.sequence import build_graph_sequence
from src.models.graph_seq import GraphSequenceModel
from src.training.graph_data import GraphSequenceDataset, collate_graph_sequence_batch
from src.training.metrics import compute_all_metrics, compute_component_cv, compute_morans_i
from src.visualization.hotspot import aggregate_attention_to_surface


TARGET_NAMES = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
COMPONENT_NAMES = {
    0: "cutterhead",
    1: "front_shield",
    2: "middle_shield",
    3: "tail_shield",
}


def save_json(data: dict[str, Any], path: Path) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert)


def build_geo_profile(tsp_df) -> tuple[np.ndarray, np.ndarray]:
    tsp_norm = normalize_coords(tsp_df)
    x = tsp_norm["X_local"].to_numpy(dtype=np.float32)
    vp = tsp_norm["Vp"].to_numpy(dtype=np.float32)
    bins = np.arange(float(x.min()), float(x.max()) + 0.5, 0.5)
    chainages, values = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.any():
            chainages.append((lo + hi) / 2)
            values.append(float(vp[mask].mean()))
    return np.asarray(chainages, dtype=np.float32), np.asarray(values, dtype=np.float32)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0, 1.0
    r, p = pearsonr(a, b)
    return float(r), float(p)


def spearman_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0, 1.0
    r, p = spearmanr(a, b)
    return float(r), float(p)


def build_test_context(cfg: dict[str, Any], device: str):
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
    attr_mean, attr_std = tsp_fit_stats_for_samples(
        tsp_df,
        sample_chainages_pre,
        train_idx,
        margin=max(float(cfg["graph"].get("tau_zone", 5.0)), float(cfg["excavation"].get("cutterhead_look_ahead", 5.0))),
    )
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
            "tsp_scaler_margin_m": max(float(cfg["graph"].get("tau_zone", 5.0)), float(cfg["excavation"].get("cutterhead_look_ahead", 5.0))),
            "tsp_attr_mean": attr_mean.tolist(),
            "tsp_attr_std": attr_std.tolist(),
        },
    }


def make_model(cfg: dict[str, Any], context: dict[str, Any], device: str) -> GraphSequenceModel:
    model_cfg = cfg["model"]
    return GraphSequenceModel(
        rock_in_dim=context["rock_in_dim"],
        tbm_in_dim=context["tbm_in_dim"],
        edge_in_dim=context["edge_in_dim"],
        monitoring_dim=6,
        output_dim=5,
        rock_hidden=model_cfg["rock_node_dim"],
        tbm_hidden=model_cfg["tbm_node_dim"],
        edge_hidden=model_cfg["edge_dim"],
        gnn_layers=model_cfg["gnn_layers"],
        gru_hidden=model_cfg["gru_hidden_dim"],
        gru_layers=model_cfg["gru_layers"],
        dropout=model_cfg["dropout"],
        residual_prediction=model_cfg.get("residual_prediction", False),
    ).to(device)


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_surface_evidence(model, context: dict[str, Any], device: str) -> dict[str, Any]:
    test_loader = context["test_loader"]
    tau = context["tau_edge"]
    attentions, edge_indices, n_tbm_list = collect_graph_attention(model, test_loader, device, tau)

    C_samples, C_sum_samples, degree_samples = [], [], []
    moran_values, cv_values = [], []
    degree_corr_values = []
    component_rows = []
    geometry_C_samples = []
    geometry_moran_values, geometry_cv_values = [], []

    for chainage, target, snap_seq, s_ij, eidx, n_tbm in zip(
        context["test_chainages"], context["y_test"], context["test_graph_seqs"], attentions, edge_indices, n_tbm_list
    ):
        snap = snap_seq[-1]
        tbm_pos = snap.hetero_data["tbm"].x[:, :3].detach().cpu().numpy()
        tbm_comp = snap.tbm_components.argmax(dim=1).detach().cpu().numpy()

        if s_ij is None or eidx is None or n_tbm == 0:
            C = np.zeros(len(tbm_pos), dtype=np.float32)
            C_sum = np.zeros(len(tbm_pos), dtype=np.float32)
        else:
            C, C_sum = aggregate_attention_to_surface(s_ij, eidx, n_tbm)
        C_samples.append(C)
        C_sum_samples.append(C_sum)

        dst = eidx[1].detach().cpu().numpy() if eidx is not None else np.array([], dtype=np.int64)
        degree = np.bincount(dst, minlength=len(tbm_pos)).astype(np.float32)
        degree_samples.append(degree)
        r_deg, _ = pearson_safe(C, degree)
        degree_corr_values.append(r_deg)

        moran_values.append(compute_morans_i(C, tbm_pos))
        cv_values.append(compute_component_cv(C, tbm_comp))
        total_relevance = float(np.sum(C) + 1e-8)
        for comp_id, comp_name in COMPONENT_NAMES.items():
            mask = tbm_comp == comp_id
            comp_sum = float(np.sum(C[mask])) if mask.any() else 0.0
            component_rows.append(
                {
                    "chainage": float(chainage),
                    "component": comp_name,
                    "mean_relevance": float(np.mean(C[mask])) if mask.any() else 0.0,
                    "sum_relevance": comp_sum,
                    "share_relevance": comp_sum / total_relevance,
                    **{name: float(value) for name, value in zip(TARGET_NAMES, target)},
                }
            )

        geometry_C = np.zeros(len(tbm_pos), dtype=np.float32)
        if ("rock", "interact", "tbm") in snap.hetero_data.edge_types:
            edge_store = snap.hetero_data["rock", "interact", "tbm"]
            geo_prior = as_numpy(edge_store["edge_attrs"]["geometry_prior"]).astype(np.float32)
            edge_dst = edge_store.edge_index[1].detach().cpu().numpy()
            geo_sum = np.zeros(len(tbm_pos), dtype=np.float32)
            geo_count = np.zeros(len(tbm_pos), dtype=np.float32)
            for val, j in zip(geo_prior, edge_dst):
                geo_sum[j] += val
                geo_count[j] += 1
            geometry_C = geo_sum / (geo_count + 1e-8)
        geometry_C_samples.append(geometry_C)
        geometry_moran_values.append(compute_morans_i(geometry_C, tbm_pos))
        geometry_cv_values.append(compute_component_cv(geometry_C, tbm_comp))

    C_mean = np.asarray([float(c.mean()) for c in C_samples], dtype=np.float32)
    geo_chainages, geo_vp = build_geo_profile(context["tsp_df"])
    test_vp = np.interp(context["test_chainages"], geo_chainages, geo_vp)
    att_geo_r, att_geo_p = pearson_safe(C_mean, test_vp)
    att_geo_sr, att_geo_sp = spearman_safe(C_mean, test_vp)

    geometry_C_mean = np.asarray([float(c.mean()) for c in geometry_C_samples], dtype=np.float32)
    geo_only_r, geo_only_p = pearson_safe(geometry_C_mean, test_vp)

    response_correlations = {}
    for j, name in enumerate(TARGET_NAMES):
        r, p = pearson_safe(C_mean, context["y_test"][:, j])
        sr, sp = spearman_safe(C_mean, context["y_test"][:, j])
        response_correlations[name] = {
            "pearson_r": r,
            "pearson_p": p,
            "spearman_r": sr,
            "spearman_p": sp,
        }

    component_summary = {}
    for comp_name in COMPONENT_NAMES.values():
        rows = [r for r in component_rows if r["component"] == comp_name]
        share = np.asarray([r["share_relevance"] for r in rows], dtype=np.float32)
        mean_rel = np.asarray([r["mean_relevance"] for r in rows], dtype=np.float32)
        component_summary[comp_name] = {
            "mean_share_relevance": float(np.mean(share)) if len(share) else 0.0,
            "std_share_relevance": float(np.std(share)) if len(share) else 0.0,
            "mean_relevance": float(np.mean(mean_rel)) if len(mean_rel) else 0.0,
        }

    return {
        "full_model": {
            "attention_geology_pearson_r": att_geo_r,
            "attention_geology_pearson_p": att_geo_p,
            "attention_geology_spearman_r": att_geo_sr,
            "attention_geology_spearman_p": att_geo_sp,
            "morans_i_mean": float(np.mean(moran_values)),
            "morans_i_std": float(np.std(moran_values)),
            "component_cv_mean": float(np.mean(cv_values)),
            "component_cv_std": float(np.std(cv_values)),
            "degree_control_pearson_r_mean": float(np.mean(degree_corr_values)),
            "degree_control_pearson_r_std": float(np.std(degree_corr_values)),
            "response_correlations": response_correlations,
            "component_summary": component_summary,
        },
        "geometry_only": {
            "attention_geology_pearson_r": geo_only_r,
            "attention_geology_pearson_p": geo_only_p,
            "morans_i_mean": float(np.mean(geometry_moran_values)),
            "morans_i_std": float(np.std(geometry_moran_values)),
            "component_cv_mean": float(np.mean(geometry_cv_values)),
            "component_cv_std": float(np.std(geometry_cv_values)),
        },
        "per_chainage": [
            {
                "chainage": float(ch),
                "mean_relevance": float(c),
                "geometry_only_mean_relevance": float(g),
                "tsp_vp": float(vp),
            }
            for ch, c, g, vp in zip(context["test_chainages"], C_mean, geometry_C_mean, test_vp)
        ],
        "component_chainage": component_rows,
    }


def write_summary_csv(evidence: dict[str, Any], path: Path) -> None:
    rows = []
    for variant in ["full_model", "geometry_only"]:
        for metric, value in evidence["surface_evidence"][variant].items():
            if isinstance(value, dict):
                continue
            rows.append({"variant": variant, "metric": metric, "value": value})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect post-hoc model evidence.")
    parser.add_argument("--config", default="config/bsll_dyk1017_205.yaml")
    parser.add_argument("--run-dir", default="outputs/bsll_dyk1017_205")
    parser.add_argument("--output-dir", default="outputs/evidence")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(__file__).resolve().parent.parent
    config_path = exp_dir / args.config
    run_dir = exp_dir / args.run_dir
    output_dir = exp_dir / args.output_dir

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    context = build_test_context(cfg, device)
    model = make_model(cfg, context, device)
    ckpt_path = run_dir / "best_graph_model.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    preds = collect_graph_predictions(model, context["test_loader"], device, context["tau_edge"])
    pred_metrics = compute_all_metrics(context["y_test"], preds)
    surface_evidence = compute_surface_evidence(model, context, device)

    evidence = {
        "config": str(config_path.relative_to(exp_dir)),
        "run_dir": str(run_dir.relative_to(exp_dir)),
        "checkpoint": str(ckpt_path.relative_to(exp_dir)),
        "split_counts": context["split_counts"],
        "preprocessing_audit": context["preprocessing_audit"],
        "prediction_metrics_from_checkpoint": pred_metrics,
        "surface_evidence": surface_evidence,
    }
    save_json(evidence, output_dir / "posthoc_evidence.json")
    write_summary_csv(evidence, output_dir / "posthoc_evidence_summary.csv")
    component_path = output_dir / "component_relevance.csv"
    with component_path.open("w", encoding="utf-8", newline="") as f:
        rows = surface_evidence["component_chainage"]
        fieldnames = list(rows[0].keys()) if rows else ["chainage", "component"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json(context["preprocessing_audit"], run_dir / "preprocessing_audit.json")

    print(f"Saved evidence to {output_dir / 'posthoc_evidence.json'}")
    print(f"Saved component table to {component_path}")
    print(
        "Full model evidence: "
        f"MAE={pred_metrics['mae']:.4f}, "
        f"Moran={surface_evidence['full_model']['morans_i_mean']:.4f}, "
        f"CV={surface_evidence['full_model']['component_cv_mean']:.4f}, "
        f"Rel-Geo r={surface_evidence['full_model']['attention_geology_pearson_r']:.4f}"
    )


if __name__ == "__main__":
    main()
