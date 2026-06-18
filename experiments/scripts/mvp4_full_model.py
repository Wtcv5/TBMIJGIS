"""MVP 4: Full experiment pipeline.

Outputs are written to ``runtime.output_dir`` from the selected YAML config.
For the current formal run, use ``config/stratified.yaml``, which writes to
``outputs/mvp4_stratified/``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.alignment import build_excavation_steps, mileage_split, stratified_split_by_geology
from src.data.monitoring import aggregate_by_chainage, load_monitoring, standardize_monitoring
from src.data.tbm_geometry import build_tbm_surface
from src.data.tsp_loader import build_voxel_field, load_tsp
from src.graph.sequence import build_graph_sequence
from src.models.baselines import LSTMBaseline, Persistence, TSPXGBoost, TSPLSTM
from src.models.graph_seq import (
    DynamicGraphOnly,
    GraphSequenceModel,
    NoGeometryPrior,
    NoGeometricConstraints,
    RandomEdgesGraphSeq,
)
from src.models.gnn import StaticGraphModel
from src.training.graph_data import GraphSequenceDataset, collate_graph_sequence_batch
from src.training.metrics import compute_all_metrics, compute_per_variable_metrics, compute_scenario_metrics, bootstrap_metrics, paired_permutation_test
from src.training.trainer import (
    StandardizedHuberLoss,
    eval_epoch,
    eval_graph_sequence_epoch,
    train_graph_sequence_model,
    train_sequence_model,
)
from src.visualization.hotspot import aggregate_attention_to_surface, plot_shield_hotspot, plot_hotspot_vs_response
from src.visualization.prediction import plot_ablation_results, plot_metrics_bar, plot_prediction_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

TARGET_NAMES = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]


# ── Helpers ────────────────────────────────────────────────────────

def build_sequence_samples(graph_snapshots, input_values, target_values, K, h):
    """Build (graph_seq, monitoring_seq) -> response samples."""
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


def collect_graph_predictions(model, dataloader, device, tau):
    """Run inference for a graph-sequence model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for graph_batch, mon_batch, _, _ in dataloader:
            pred, _ = model(graph_batch, mon_batch.to(device), tau=tau)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def collect_graph_attention(model, dataloader, device, tau):
    """Extract per-sample attention weights from the last graph snapshot."""
    model.eval()
    all_attentions = []
    all_edge_indices = []
    all_n_tbm = []
    with torch.no_grad():
        for graph_batch, mon_batch, _, _ in dataloader:
            _, attentions = model(graph_batch, mon_batch.to(device), tau=tau,
                                  return_attention=True)
            # attentions: list of s_ij (raw scores, pre-softmax) per sample (last snapshot)
            for i, snap_seq in enumerate(graph_batch):
                last_snap = snap_seq[-1]
                if ("rock", "interact", "tbm") in last_snap.hetero_data.edge_types:
                    edge_idx = last_snap.hetero_data["rock", "interact", "tbm"].edge_index
                    n_tbm = last_snap.hetero_data["tbm"].x.shape[0]
                    s_ij = attentions[i] if attentions and attentions[i] is not None else None
                    all_attentions.append(s_ij)
                    all_edge_indices.append(edge_idx)
                    all_n_tbm.append(n_tbm)
                else:
                    all_attentions.append(None)
                    all_edge_indices.append(None)
                    all_n_tbm.append(0)
    return all_attentions, all_edge_indices, all_n_tbm


def compute_tsp_stats_per_sample(graph_seqs, K):
    """Compute TSP active-zone statistics for each sample (for XGBoost/TSP-LSTM)."""
    stats_list = []
    for snap_seq in graph_seqs:
        # Use the last snapshot's rock attributes as TSP statistics
        last_snap = snap_seq[-1]
        rock_attrs = last_snap.rock_attrs.cpu().numpy()
        if len(rock_attrs) > 0:
            stats = np.concatenate([
                rock_attrs.mean(axis=0),
                rock_attrs.std(axis=0),
                rock_attrs.min(axis=0),
                rock_attrs.max(axis=0),
            ])
        else:
            stats = np.zeros(rock_attrs.shape[1] * 4 if len(rock_attrs.shape) > 1 else 8)
        stats_list.append(stats)
    return np.stack(stats_list)


def save_json(data, path):
    """Save dict to JSON with numpy conversion."""
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else convert(v)
                   for k, v in data.items()}, f, indent=2, ensure_ascii=False)


# ── Main ───────────────────────────────────────────────────────────

def main(config_path: str = "config/default.yaml"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    runtime_cfg = cfg.get("runtime", {})

    # Fix random seeds for reproducibility
    SEED = cfg.get("seed", 42)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} (seed={SEED})")

    output_dir = Path(runtime_cfg.get("output_dir", "outputs/mvp4"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loading ───────────────────────────────────────────────
    logger.info("Loading data...")
    tsp_df = load_tsp(cfg["data"]["tsp_path"])
    mon_df = load_monitoring(cfg["data"]["monitoring_path"])
    mon_df = aggregate_by_chainage(mon_df, cfg["excavation"]["step_length"])

    # Load augmented data for training (if enabled)
    # Augmented data is pre-aggregated in generate_augmented_data.py, so we skip
    # aggregate_by_chainage here to preserve decimal chainages of augmented samples.
    use_augmentation = runtime_cfg.get("use_augmentation", False)
    aug_mon_df = None
    if use_augmentation:
        aug_path = cfg["data"].get("monitoring_augmented_path",
                                    "data/raw/monitoring_augmented.csv")
        if Path(aug_path).exists():
            aug_mon_df = load_monitoring(aug_path)
            # Do NOT aggregate — augmented data already at 1m resolution with decimal offsets
            logger.info(f"Loaded augmented data: {len(aug_mon_df)} samples")
        else:
            logger.warning(f"Augmented data not found at {aug_path}, using original only")
            use_augmentation = False

    K = cfg["model"]["history_window"]
    h = cfg["model"]["predict_horizon"]
    n_total = len(mon_df) - K - h + 1
    if n_total <= 0:
        raise ValueError(
            f"Not enough monitoring rows ({len(mon_df)}) for K={K}, h={h}."
        )

    # ── Data split ─────────────────────────────────────────────────
    split_method = cfg.get("split", {}).get("method", "mileage")
    if split_method == "stratified":
        logger.info("Using stratified split by geology (Vs)...")
        train_idx, val_idx, test_idx = stratified_split_by_geology(
            mon_df, tsp_df, n_total, K=K, h=h,
            train_r=cfg["split"]["train_ratio"],
            val_r=cfg["split"]["val_ratio"],
            n_strata=cfg.get("split", {}).get("n_strata", 4),
            seed=SEED,
        )
        train_fit_stop = int(max(train_idx)) + K + h
        logger.info(f"Stratified split — Train: {len(train_idx)}, "
                    f"Val: {len(val_idx)}, Test: {len(test_idx)}")
    else:
        train_slice, val_slice, test_slice = mileage_split(n_total)
        train_idx = np.arange(train_slice.start, train_slice.stop)
        val_idx = np.arange(val_slice.start, val_slice.stop)
        test_idx = np.arange(test_slice.start, test_slice.stop)
        train_fit_stop = train_slice.stop + K + h - 1
        logger.info(f"Mileage split — Train: {len(train_idx)}, "
                    f"Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 计算训练集 TSP 属性统计量，避免数据泄漏
    from src.data.tsp_loader import TSP_ATTR_COLS
    tsp_raw = tsp_df[TSP_ATTR_COLS].to_numpy(dtype=np.float32)
    # 训练集对应的 TSP 体素: 使用全部 TSP 数据的统计量
    # (TSP 是地质先验，不随时间变化，但标准化应仅用训练区间内的统计量)
    # 简化处理: 使用全部 TSP 数据计算统计量 (地质属性是静态的)
    # 但严格来说应只用训练区间内的体素
    attr_mean = tsp_raw.mean(axis=0, keepdims=True)
    attr_std = tsp_raw.std(axis=0, keepdims=True) + 1e-8

    rock_coords, rock_attrs = build_voxel_field(tsp_df, attr_mean=attr_mean, attr_std=attr_std)

    mon_df, scaler = standardize_monitoring(
        mon_df, fit_df=mon_df.iloc[:train_fit_stop]
    )

    # Standardize augmented data using the SAME scaler (fit on original train only)
    if use_augmentation and aug_mon_df is not None:
        aug_mon_df, _ = standardize_monitoring(aug_mon_df, fit_df=mon_df.iloc[:train_fit_stop].copy())
        # Note: aug_mon_df is standardized with original train stats; we'll use it
        # only for building training samples. The original mon_df is used for val/test.

    tbm_cfg = cfg["tbm_geometry"]
    tbm_surface = build_tbm_surface(
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        front_len=tbm_cfg["front_shield_len"],
        middle_len=tbm_cfg["middle_shield_len"],
        tail_len=tbm_cfg["tail_shield_len"],
        resolution=tbm_cfg["surface_resolution"],
    )

    tau_zone = cfg["graph"].get("tau_zone", 5.0)
    tau_edge = cfg["graph"].get("tau_edge", cfg["graph"].get("distance_threshold", 2.0))
    eta_min = cfg["graph"]["normal_threshold"]
    # Alias for model/training functions that use `tau` as the edge distance threshold
    tau = tau_edge

    steps = build_excavation_steps(
        mon_df,
        step_length=cfg["excavation"]["step_length"],
        K=K, h=h,
    )
    logger.info(f"Building graph sequence ({len(steps)} steps)...")
    snapshots = build_graph_sequence(
        steps, rock_coords, rock_attrs, tbm_surface,
        cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
        shield_radius=tbm_cfg["shield_radius"],
        tau_zone=tau_zone, tau_edge=tau_edge, eta_min=eta_min,
        device=device, verbose=True,
    )

    # NOTE: NoGeometricConstraints ablation (A4) requires building a relaxed graph
    # with larger tau and eta_min=0.0, which may cause OOM on limited hardware.
    # To enable A4, set runtime.enable_a4=true in config.
    enable_a4 = runtime_cfg.get("enable_a4", False)
    relaxed_snapshots = None
    if enable_a4:
        logger.info("Building relaxed graph sequence for A4 ablation...")
        relaxed_tau_edge = 5.0
        relaxed_snapshots = build_graph_sequence(
            steps, rock_coords, rock_attrs, tbm_surface,
            cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
            shield_radius=tbm_cfg["shield_radius"],
            tau_zone=relaxed_tau_edge, tau_edge=relaxed_tau_edge, eta_min=0.0,
            device=device, verbose=True,
        )

    mon_features = mon_df[
        ["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]
    ].to_numpy(dtype=np.float32)
    mon_targets = mon_df[
        ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
    ].to_numpy(dtype=np.float32)

    graph_seqs, X_mon, y, sample_chainages = build_sequence_samples(
        snapshots, mon_features, mon_targets, K, h
    )
    relaxed_graph_seqs = None
    if len(y) != n_total:
        raise RuntimeError(
            f"Sample count mismatch: expected {n_total}, got {len(y)}."
        )

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)
    logger.info(f"Split — Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # ── Build augmented training samples (if enabled) ──────────────
    # Augmented samples are ONLY used for training; val/test use original data.
    aug_X_mon_train = None
    aug_y_train = None
    aug_graph_seqs_train = None
    if use_augmentation and aug_mon_df is not None:
        logger.info("Building augmented training samples...")
        # Use augmented data only within the training chainage range
        # (chainage 0 to train_fit_stop in original indexing)
        train_chainage_max = mon_df.iloc[train_fit_stop - 1]["chainage"]
        aug_train_df = aug_mon_df[aug_mon_df["chainage"] <= train_chainage_max].copy()

        if len(aug_train_df) > n_train + K + h - 1:
            aug_steps = build_excavation_steps(
                aug_train_df,
                step_length=cfg["excavation"]["step_length"],
                K=K, h=h,
            )
            aug_snapshots = build_graph_sequence(
                aug_steps, rock_coords, rock_attrs, tbm_surface,
                cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
                shield_radius=tbm_cfg["shield_radius"],
                tau_zone=tau_zone, tau_edge=tau_edge, eta_min=eta_min,
                device=device, verbose=False,
            )
            aug_mon_features = aug_train_df[
                ["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]
            ].to_numpy(dtype=np.float32)
            aug_mon_targets = aug_train_df[
                ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
            ].to_numpy(dtype=np.float32)
            aug_graph_seqs_train, aug_X_mon_train, aug_y_train, _ = build_sequence_samples(
                aug_snapshots, aug_mon_features, aug_mon_targets, K, h
            )
            logger.info(f"Augmented training samples: {len(aug_y_train)} "
                        f"(original train: {n_train})")

    # ── Dataset / DataLoader setup ─────────────────────────────────
    X_mon_t = torch.tensor(X_mon, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Sequence datasets (LSTM, TSP-LSTM)
    # For LSTM/TSP-LSTM: use augmented training data if available
    if use_augmentation and aug_X_mon_train is not None and len(aug_X_mon_train) > 0:
        train_ds = TensorDataset(
            torch.tensor(aug_X_mon_train, dtype=torch.float32),
            torch.tensor(aug_y_train, dtype=torch.float32),
        )
        logger.info(f"LSTM/TSP-LSTM training on augmented data: {len(aug_X_mon_train)} samples")
    else:
        train_ds = TensorDataset(X_mon_t[train_idx], y_t[train_idx])
    val_ds = TensorDataset(X_mon_t[val_idx], y_t[val_idx])
    test_ds = TensorDataset(X_mon_t[test_idx], y_t[test_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])

    # Graph sequence datasets
    def make_graph_loaders(full_graph_seqs):
        """Create train/val/test loaders from the full graph sequence list."""
        # For training: use augmented graph sequences if available
        if use_augmentation and aug_graph_seqs_train is not None and len(aug_graph_seqs_train) > 0:
            train_gds = GraphSequenceDataset(
                aug_graph_seqs_train, aug_X_mon_train, aug_y_train,
                np.zeros(len(aug_y_train), dtype=np.float32),  # placeholder chainages
            )
            logger.info(f"Graph model training on augmented data: {len(aug_graph_seqs_train)} samples")
        else:
            train_gds = GraphSequenceDataset(
                [full_graph_seqs[i] for i in train_idx], X_mon[train_idx], y[train_idx], sample_chainages[train_idx]
            )
        val_gds = GraphSequenceDataset(
            [full_graph_seqs[i] for i in val_idx], X_mon[val_idx], y[val_idx], sample_chainages[val_idx]
        )
        test_gds = GraphSequenceDataset(
            [full_graph_seqs[i] for i in test_idx], X_mon[test_idx], y[test_idx], sample_chainages[test_idx]
        )
        train_gl = DataLoader(train_gds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_graph_sequence_batch)
        val_gl = DataLoader(val_gds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, collate_fn=collate_graph_sequence_batch)
        test_gl = DataLoader(test_gds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, collate_fn=collate_graph_sequence_batch)
        return train_gl, val_gl, test_gl

    graph_train_loader, graph_val_loader, graph_test_loader = make_graph_loaders(
        graph_seqs
    )
    # Relaxed graph loaders — only available if relaxed_snapshots was built
    relaxed_train_loader = relaxed_val_loader = relaxed_test_loader = None
    if relaxed_snapshots is not None:
        relaxed_graph_seqs, relaxed_X_mon, relaxed_y, relaxed_chainages = build_sequence_samples(
            relaxed_snapshots, mon_features, mon_targets, K, h
        )
        if len(relaxed_y) == n_total:
            relaxed_train_gds = GraphSequenceDataset(
                [relaxed_graph_seqs[i] for i in train_idx], X_mon[train_idx], y[train_idx], sample_chainages[train_idx]
            )
            relaxed_val_gds = GraphSequenceDataset(
                [relaxed_graph_seqs[i] for i in val_idx], X_mon[val_idx], y[val_idx], sample_chainages[val_idx]
            )
            relaxed_test_gds = GraphSequenceDataset(
                [relaxed_graph_seqs[i] for i in test_idx], X_mon[test_idx], y[test_idx], sample_chainages[test_idx]
            )
            relaxed_train_loader = DataLoader(relaxed_train_gds, batch_size=cfg["training"]["batch_size"],
                                              shuffle=False, collate_fn=collate_graph_sequence_batch)
            relaxed_val_loader = DataLoader(relaxed_val_gds, batch_size=cfg["training"]["batch_size"],
                                            shuffle=False, collate_fn=collate_graph_sequence_batch)
            relaxed_test_loader = DataLoader(relaxed_test_gds, batch_size=cfg["training"]["batch_size"],
                                             shuffle=False, collate_fn=collate_graph_sequence_batch)

    # TSP statistics for XGBoost / TSP-LSTM
    tsp_stats = compute_tsp_stats_per_sample(graph_seqs, K)

    # Model dimensions
    rock_in_dim = rock_coords.shape[1] + rock_attrs.shape[1] + 1
    tbm_in_dim = 3 + 3 + 4
    edge_in_dim = 1 + 1 + 3 + rock_attrs.shape[1] + 4

    train_kwargs = dict(
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        huber_delta=cfg["training"]["huber_delta"],
        device=device,
    )

    # ── 1. Persistence baseline ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("1. Persistence baseline")
    persist = Persistence()
    target_feature_idx = [0, 2, 3, 4, 5]
    persist_preds = X_mon[test_idx][:, -1, :][:, target_feature_idx]
    persist_metrics = compute_all_metrics(y[test_idx], persist_preds)
    persist_per_var = compute_per_variable_metrics(y[test_idx], persist_preds, TARGET_NAMES)
    logger.info(f"  Persistence MAE={persist_metrics['mae']:.4f} RMSE={persist_metrics['rmse']:.4f} R2={persist_metrics['r2']:.4f}")

    # ── 2. XGBoost baseline ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("2. XGBoost + TSP baseline")
    xgb = TSPXGBoost(
        n_estimators=runtime_cfg.get("xgb_n_estimators", 200),
        max_depth=runtime_cfg.get("xgb_max_depth", 6),
    )
    xgb.fit(X_mon[train_idx], tsp_stats[train_idx], y[train_idx])
    xgb_preds = xgb.predict(X_mon[test_idx], tsp_stats[test_idx])
    xgb_metrics = compute_all_metrics(y[test_idx], xgb_preds)
    xgb_per_var = compute_per_variable_metrics(y[test_idx], xgb_preds, TARGET_NAMES)
    logger.info(f"  XGBoost MAE={xgb_metrics['mae']:.4f} RMSE={xgb_metrics['rmse']:.4f} R2={xgb_metrics['r2']:.4f}")

    # ── 3. LSTM baseline ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("3. LSTM baseline")
    lstm = LSTMBaseline(input_dim=6, hidden_dim=128, num_layers=2,
                        output_dim=5, dropout=0.1).to(device)
    lstm, _ = train_sequence_model(lstm, train_loader, val_loader,
                                   checkpoint_dir=output_dir, **train_kwargs)
    lstm_results = eval_epoch(lstm, test_loader, StandardizedHuberLoss(5), device)
    lstm.eval()
    with torch.no_grad():
        lstm_preds = lstm(X_mon_t[test_idx].to(device)).cpu().numpy()
    lstm_per_var = compute_per_variable_metrics(y[test_idx], lstm_preds, TARGET_NAMES)
    logger.info(f"  LSTM MAE={lstm_results['mae']:.4f} RMSE={lstm_results['rmse']:.4f} R2={lstm_results['r2']:.4f}")

    # ── 4. TSP-LSTM baseline ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("4. TSP-LSTM baseline")
    tsp_stat_dim = tsp_stats.shape[1]
    tsp_lstm = TSPLSTM(input_dim=6, tsp_stat_dim=tsp_stat_dim,
                       hidden_dim=128, num_layers=2, output_dim=5, dropout=0.1).to(device)

    # For TSP-LSTM: use augmented training data if available
    if use_augmentation and aug_X_mon_train is not None and len(aug_X_mon_train) > 0:
        # Compute TSP stats for augmented samples by interpolating from original
        # Use nearest original sample's TSP stats (simplified approach)
        n_aug_train = len(aug_X_mon_train)
        # Repeat original train TSP stats to match augmented samples (approximation)
        # This is a simplification; ideally we'd recompute TSP stats for each augmented sample
        aug_tsp_stats = np.tile(tsp_stats[train_idx], (n_aug_train // n_train + 1, 1))[:n_aug_train]
        # Add small noise to TSP stats for augmented samples
        rng = np.random.RandomState(SEED)
        aug_tsp_stats = aug_tsp_stats + rng.normal(0, 0.01, aug_tsp_stats.shape).astype(np.float32)

        tsp_train_ds = TensorDataset(
            torch.tensor(aug_X_mon_train, dtype=torch.float32),
            torch.tensor(aug_tsp_stats, dtype=torch.float32),
            torch.tensor(aug_y_train, dtype=torch.float32),
        )
        logger.info(f"TSP-LSTM training on augmented data: {n_aug_train} samples")
    else:
        tsp_train_ds = TensorDataset(
            X_mon_t[train_idx],
            torch.tensor(tsp_stats[train_idx], dtype=torch.float32),
            y_t[train_idx],
        )
    tsp_val_ds = TensorDataset(
        X_mon_t[val_idx],
        torch.tensor(tsp_stats[val_idx], dtype=torch.float32),
        y_t[val_idx],
    )
    tsp_test_ds = TensorDataset(
        X_mon_t[test_idx],
        torch.tensor(tsp_stats[test_idx], dtype=torch.float32),
        y_t[test_idx],
    )
    tsp_train_loader = DataLoader(tsp_train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    tsp_val_loader = DataLoader(tsp_val_ds, batch_size=cfg["training"]["batch_size"])
    tsp_test_loader = DataLoader(tsp_test_ds, batch_size=cfg["training"]["batch_size"])

    loss_fn = StandardizedHuberLoss(n_targets=5, delta=train_kwargs["huber_delta"])
    optimizer = torch.optim.AdamW(tsp_lstm.parameters(), lr=train_kwargs["lr"],
                                  weight_decay=train_kwargs["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    from src.training.trainer import EarlyStopping
    es = EarlyStopping(patience=train_kwargs["patience"])
    best_state = None

    for epoch in range(1, train_kwargs["epochs"] + 1):
        tsp_lstm.train()
        for x_seq, tsp_s, yt in tsp_train_loader:
            x_seq, tsp_s, yt = x_seq.to(device), tsp_s.to(device), yt.to(device)
            optimizer.zero_grad()
            pred = tsp_lstm(x_seq, tsp_s)
            loss = loss_fn(pred, yt)
            loss.backward()
            optimizer.step()

        tsp_lstm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, tsp_s, yt in tsp_val_loader:
                pred = tsp_lstm(x_seq.to(device), tsp_s.to(device))
                val_loss += loss_fn(pred, yt.to(device)).item()
        val_loss /= len(tsp_val_loader)
        scheduler.step(val_loss)
        if es.update(val_loss):
            best_state = {k: v.cpu().clone() for k, v in tsp_lstm.state_dict().items()}
        if es.should_stop:
            logger.info(f"  TSP-LSTM early stop at epoch {epoch}")
            break

    if best_state:
        tsp_lstm.load_state_dict(best_state)

    tsp_lstm.eval()
    with torch.no_grad():
        tsp_lstm_preds = tsp_lstm(
            X_mon_t[test_idx].to(device),
            torch.tensor(tsp_stats[test_idx], dtype=torch.float32).to(device),
        ).cpu().numpy()
    tsp_lstm_metrics = compute_all_metrics(y[test_idx], tsp_lstm_preds)
    tsp_lstm_per_var = compute_per_variable_metrics(y[test_idx], tsp_lstm_preds, TARGET_NAMES)
    logger.info(f"  TSP-LSTM MAE={tsp_lstm_metrics['mae']:.4f} RMSE={tsp_lstm_metrics['rmse']:.4f} R2={tsp_lstm_metrics['r2']:.4f}")

    # ── 5. Full Model (GNN + GRU) ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("5. Full Model (GNN + GRU)")

    def make_full_model():
        return GraphSequenceModel(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            monitoring_dim=6, output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"],
            tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"],
            gnn_layers=cfg["model"]["gnn_layers"],
            gru_hidden=cfg["model"]["gru_hidden_dim"],
            gru_layers=cfg["model"]["gru_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)

    full_model = make_full_model()
    logger.info(f"  Model params: {sum(p.numel() for p in full_model.parameters()):,}")
    full_model, _ = train_graph_sequence_model(
        full_model, graph_train_loader, graph_val_loader,
        tau=tau, checkpoint_dir=output_dir, **train_kwargs,
    )
    full_results = eval_graph_sequence_epoch(
        full_model, graph_test_loader, StandardizedHuberLoss(5), device, tau=tau
    )
    full_preds = collect_graph_predictions(full_model, graph_test_loader, device, tau)
    full_per_var = compute_per_variable_metrics(y[test_idx], full_preds, TARGET_NAMES)
    logger.info(f"  Full Model MAE={full_results['mae']:.4f} RMSE={full_results['rmse']:.4f} R2={full_results['r2']:.4f}")

    # ── Ablation experiments ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Ablation experiments")

    ablation_results = {"Full Model": full_results}
    ablation_preds = {"Full Model": full_preds}

    if runtime_cfg.get("run_ablations", True):
        # A1: No monitoring input (graph only)
        logger.info("  A1: No monitoring input")
        graph_only = DynamicGraphOnly(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            monitoring_dim=6, output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"],
            tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"],
            gnn_layers=cfg["model"]["gnn_layers"],
            gru_hidden=cfg["model"]["gru_hidden_dim"],
            gru_layers=cfg["model"]["gru_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        graph_only, _ = train_graph_sequence_model(
            graph_only, graph_train_loader, graph_val_loader,
            tau=tau, checkpoint_dir=output_dir, checkpoint_name="ablation_no_mon.pt", **train_kwargs,
        )
        a1_results = eval_graph_sequence_epoch(graph_only, graph_test_loader,
                                               StandardizedHuberLoss(5), device, tau=tau)
        a1_preds = collect_graph_predictions(graph_only, graph_test_loader, device, tau)
        ablation_results["No Monitoring"] = a1_results
        ablation_preds["No Monitoring"] = a1_preds
        logger.info(f"    MAE={a1_results['mae']:.4f} RMSE={a1_results['rmse']:.4f}")

        # A2: Randomized rock-machine edges
        logger.info("  A2: Randomized edges")
        rand_model = RandomEdgesGraphSeq(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            monitoring_dim=6, output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"],
            tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"],
            gnn_layers=cfg["model"]["gnn_layers"],
            gru_hidden=cfg["model"]["gru_hidden_dim"],
            gru_layers=cfg["model"]["gru_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        rand_model, _ = train_graph_sequence_model(
            rand_model, graph_train_loader, graph_val_loader,
            tau=tau, checkpoint_dir=output_dir, checkpoint_name="ablation_random_edges.pt", **train_kwargs,
        )
        a2_results = eval_graph_sequence_epoch(rand_model, graph_test_loader,
                                               StandardizedHuberLoss(5), device, tau=tau)
        a2_preds = collect_graph_predictions(rand_model, graph_test_loader, device, tau)
        ablation_results["Random Edges"] = a2_results
        ablation_preds["Random Edges"] = a2_preds
        logger.info(f"    MAE={a2_results['mae']:.4f} RMSE={a2_results['rmse']:.4f}")

        # A3: No geometric prior (β=0)
        logger.info("  A3: No geometric prior")
        no_prior = NoGeometryPrior(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            monitoring_dim=6, output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"],
            tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"],
            gnn_layers=cfg["model"]["gnn_layers"],
            gru_hidden=cfg["model"]["gru_hidden_dim"],
            gru_layers=cfg["model"]["gru_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        no_prior, _ = train_graph_sequence_model(
            no_prior, graph_train_loader, graph_val_loader,
            tau=tau, checkpoint_dir=output_dir, checkpoint_name="ablation_no_prior.pt", **train_kwargs,
        )
        a3_results = eval_graph_sequence_epoch(no_prior, graph_test_loader,
                                               StandardizedHuberLoss(5), device, tau=tau)
        a3_preds = collect_graph_predictions(no_prior, graph_test_loader, device, tau)
        ablation_results["No Geometry Prior"] = a3_results
        ablation_preds["No Geometry Prior"] = a3_preds
        logger.info(f"    MAE={a3_results['mae']:.4f} RMSE={a3_results['rmse']:.4f}")

        # A4: No geometric constraints (relaxed graph) — requires relaxed graph
        if relaxed_train_loader is not None:
            logger.info("  A4: No geometric constraints (relaxed graph)...")
            try:
                no_constraints = NoGeometricConstraints(
                    rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
                    monitoring_dim=6, output_dim=5,
                    rock_hidden=cfg["model"]["rock_node_dim"],
                    tbm_hidden=cfg["model"]["tbm_node_dim"],
                    edge_hidden=cfg["model"]["edge_dim"],
                    gnn_layers=cfg["model"]["gnn_layers"],
                    gru_hidden=cfg["model"]["gru_hidden_dim"],
                    gru_layers=cfg["model"]["gru_layers"],
                    dropout=cfg["model"]["dropout"],
                ).to(device)
                no_constraints, _ = train_graph_sequence_model(
                    no_constraints, relaxed_train_loader, relaxed_val_loader,
                    tau=5.0, checkpoint_dir=output_dir, checkpoint_name="ablation_no_constraints.pt", **train_kwargs,
                )
                a4_results = eval_graph_sequence_epoch(no_constraints, relaxed_test_loader,
                                                       StandardizedHuberLoss(5), device, tau=5.0)
                a4_preds = collect_graph_predictions(no_constraints, relaxed_test_loader, device, tau=5.0)
                ablation_results["No Geometric Constraints"] = a4_results
                ablation_preds["No Geometric Constraints"] = a4_preds
                logger.info(f"    MAE={a4_results['mae']:.4f} RMSE={a4_results['rmse']:.4f}")
            except (RuntimeError, MemoryError) as e:
                logger.warning(f"    A4 skipped due to: {e}")
        else:
            logger.info("  A4: Skipped (relaxed graph not available)")
    else:
        logger.info("  Ablations skipped by runtime.run_ablations=false")

    # ── Attention extraction & hotspot generation ──────────────────
    logger.info("=" * 60)
    logger.info("Extracting attention and generating hotspot maps...")
    attentions, edge_indices, n_tbm_list = collect_graph_attention(
        full_model, graph_test_loader, device, tau
    )

    # Aggregate attention per sample and generate hotspot maps for selected samples
    C_j_samples = []
    for i, (alpha, eidx, n_tbm) in enumerate(zip(attentions, edge_indices, n_tbm_list)):
        if alpha is not None and eidx is not None:
            C_j = aggregate_attention_to_surface(alpha, eidx, n_tbm)
            C_j_samples.append(C_j)
        else:
            C_j_samples.append(None)

    # Generate hotspot figures for a few representative chainages
    test_chainages = sample_chainages[test_idx]
    test_graph_seqs = [graph_seqs[i] for i in test_idx]
    n_hotspot_samples = min(5, len(C_j_samples))
    step_indices = np.linspace(0, len(C_j_samples) - 1, n_hotspot_samples, dtype=int)

    for idx in step_indices:
        if C_j_samples[idx] is None:
            continue
        # Get the TBM surface for this snapshot
        snap = test_graph_seqs[idx][-1]
        tbm_pos = snap.hetero_data["tbm"].x[:, :3].cpu().numpy()
        # Reconstruct components from one-hot
        tbm_comp = snap.tbm_components.argmax(dim=1).cpu().numpy()
        chainage_val = float(test_chainages[idx])

        plot_shield_hotspot(
            tbm_pos, tbm_comp, C_j_samples[idx],
            save_path=str(output_dir / f"hotspot_chainage_{chainage_val:.0f}.pdf"),
        )

    # Chainage evolution plot
    C_mean_list = []
    for C_j in C_j_samples:
        if C_j is not None and len(C_j) > 0:
            C_j_arr = np.asarray(C_j)
            if C_j_arr.ndim > 1:
                C_j_arr = C_j_arr[-1]
            C_mean_list.append(float(C_j_arr.mean()))
        else:
            C_mean_list.append(0.0)
    C_mean_arr = np.array(C_mean_list)

    plot_hotspot_vs_response(
        test_chainages.tolist(), C_mean_arr, y[test_idx], TARGET_NAMES,
        save_path=str(output_dir / "chainage_evolution.pdf"),
    )

    # ── Scenario evaluation ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Computing scenario metrics...")
    test_mon = y[test_idx]
    scenario_masks = {}
    for var_name, pct, label in [
        ("Thrust", 80, "High Thrust"),
        ("Torque", 80, "High Torque"),
        ("Penetration", 20, "Low Penetration"),
        ("AdvanceRate", 20, "Low Advance Rate"),
    ]:
        var_idx = TARGET_NAMES.index(var_name)
        if "Low" in label:
            threshold = np.percentile(test_mon[:, var_idx], pct)
            mask = test_mon[:, var_idx] <= threshold
        else:
            threshold = np.percentile(test_mon[:, var_idx], pct)
            mask = test_mon[:, var_idx] >= threshold
        scenario_masks[label] = mask

    scenario_metrics = {}
    for model_name, preds in [
        ("Persistence", persist_preds),
        ("XGBoost", xgb_preds),
        ("LSTM", lstm_preds),
        ("TSP-LSTM", tsp_lstm_preds),
        ("Full Model", full_preds),
    ]:
        scenario_metrics[model_name] = compute_scenario_metrics(
            y[test_idx], preds, test_mon, scenario_masks
        )

    # ── Bootstrap confidence intervals ─────────────────────────────
    logger.info("=" * 60)
    logger.info("Computing bootstrap confidence intervals...")

    all_preds_map = {
        "Persistence": persist_preds,
        "XGBoost": xgb_preds,
        "LSTM": lstm_preds,
        "TSP-LSTM": tsp_lstm_preds,
        "Full Model": full_preds,
    }
    bootstrap_results = {}
    n_boot = int(runtime_cfg.get("bootstrap_samples", 1000))
    if n_boot > 0:
        for name, preds in all_preds_map.items():
            bootstrap_results[name] = bootstrap_metrics(y[test_idx], preds, n_boot=n_boot)
            logger.info(f"  {name}: MAE={bootstrap_results[name]['mae']['point']:.4f} "
                        f"[{bootstrap_results[name]['mae']['ci_low']:.4f}, "
                        f"{bootstrap_results[name]['mae']['ci_high']:.4f}]")
    else:
        logger.info("  Bootstrap skipped by runtime.bootstrap_samples=0")

    # ── Paired permutation tests (Full Model vs each baseline) ────
    logger.info("=" * 60)
    logger.info("Paired permutation tests (Full Model vs baselines)...")
    significance_results = {}
    n_perm = int(runtime_cfg.get("permutation_samples", 10000))
    if n_perm > 0:
        for name, preds in all_preds_map.items():
            if name == "Full Model":
                continue
            delta_mae, p_mae = paired_permutation_test(
                y[test_idx], full_preds, preds, metric="mae", n_perm=n_perm)
            delta_rmse, p_rmse = paired_permutation_test(
                y[test_idx], full_preds, preds, metric="rmse", n_perm=n_perm)
            significance_results[name] = {
                "delta_mae": delta_mae, "p_mae": p_mae,
                "delta_rmse": delta_rmse, "p_rmse": p_rmse,
            }
            sig_mae = "*" if p_mae < 0.05 else ""
            sig_rmse = "*" if p_rmse < 0.05 else ""
            logger.info(f"  vs {name}: ΔMAE={delta_mae:.4f} (p={p_mae:.4f}){sig_mae}  "
                        f"ΔRMSE={delta_rmse:.4f} (p={p_rmse:.4f}){sig_rmse}")
    else:
        logger.info("  Permutation tests skipped by runtime.permutation_samples=0")

    # ── Per-variable ablation comparison ──────────────────────────
    logger.info("=" * 60)
    logger.info("Per-variable ablation metrics...")
    ablation_per_var = {}
    for name, preds in ablation_preds.items():
        ablation_per_var[name] = compute_per_variable_metrics(y[test_idx], preds, TARGET_NAMES)

    # ── Save all metrics ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Saving metrics...")

    global_metrics = {
        "Persistence": persist_metrics,
        "XGBoost": xgb_metrics,
        "LSTM": lstm_results,
        "TSP-LSTM": tsp_lstm_metrics,
        "Full Model": full_results,
    }
    save_json(global_metrics, output_dir / "metrics_global.json")

    per_var_metrics = {
        "Persistence": persist_per_var,
        "XGBoost": xgb_per_var,
        "LSTM": lstm_per_var,
        "TSP-LSTM": tsp_lstm_per_var,
        "Full Model": full_per_var,
    }
    save_json(per_var_metrics, output_dir / "metrics_per_variable.json")

    save_json(scenario_metrics, output_dir / "metrics_scenario.json")
    save_json(ablation_results, output_dir / "ablation_metrics.json")
    save_json(bootstrap_results, output_dir / "bootstrap_ci.json")
    save_json(significance_results, output_dir / "significance_tests.json")
    save_json(ablation_per_var, output_dir / "ablation_per_variable.json")

    # ── Generate figures ───────────────────────────────────────────
    logger.info("Generating figures...")

    # Per-variable prediction comparison plots
    for vi, vname in enumerate(TARGET_NAMES):
        plot_prediction_comparison(
            test_chainages, y[test_idx],
            {"Full Model": full_preds, "LSTM": lstm_preds, "TSP-LSTM": tsp_lstm_preds,
             "XGBoost": xgb_preds, "Persistence": persist_preds},
            var_idx=vi, var_name=vname,
            save_path=str(output_dir / f"pred_comparison_{vname}.pdf"),
        )

    plot_metrics_bar(global_metrics, metric_name="rmse",
                     save_path=str(output_dir / "metrics_comparison_rmse.pdf"))
    plot_metrics_bar(global_metrics, metric_name="mae",
                     save_path=str(output_dir / "metrics_comparison_mae.pdf"))

    plot_ablation_results(ablation_results, metric_name="rmse",
                          save_path=str(output_dir / "ablation_comparison.pdf"))

    # ── Print summary table ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY TABLE — Global Metrics")
    logger.info(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Pearson':>8}")
    for name, m in global_metrics.items():
        logger.info(f"{name:<25} {m['mae']:8.4f} {m['rmse']:8.4f} {m['r2']:8.4f} {m['pearson']:8.4f}")

    logger.info("\nABLATION TABLE")
    logger.info(f"{'Variant':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    for name, m in ablation_results.items():
        logger.info(f"{name:<30} {m['mae']:8.4f} {m['rmse']:8.4f} {m['r2']:8.4f}")

    logger.info(f"\nAll outputs saved to {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MVP4 experiment pipeline.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)
