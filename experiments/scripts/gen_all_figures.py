"""Generate ALL publication figures for the IJGIS manuscript.

Produces the following figures:
  Fig 1: Framework overview pipeline (schematic)
  Fig 2: Graph construction 4-panel (schematic + adjacency)
  Fig 3: Prediction comparison (per-variable, 5 panels)
  Fig 4: Ablation study (grouped bar chart)
  Fig 5: Hotspot maps (unwrapped surface, 3 chainages)
  Fig 6: Chainage evolution (stacked panels)

Usage:
  python scripts/gen_all_figures.py [--config config/default.yaml]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import torch
import yaml
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader, TensorDataset

from src.data.alignment import build_excavation_steps, mileage_split
from src.data.monitoring import aggregate_by_chainage, load_monitoring, standardize_monitoring
from src.data.tbm_geometry import build_tbm_surface
from src.data.tsp_loader import build_voxel_field, load_tsp, normalize_coords, get_rock_coords, get_rock_attrs, TSP_ATTR_COLS
from src.graph.sequence import build_graph_sequence
from src.models.baselines import LSTMBaseline, Persistence, TSPXGBoost, TSPLSTM, SpatialAggLSTM, compute_spatial_agg_features
from src.models.graph_seq import (
    DynamicGraphOnly, GraphSequenceModel, NoGeometryPrior, RandomEdgesGraphSeq,
)
from src.models.gnn import StaticGraphModel
from src.training.graph_data import GraphSequenceDataset, collate_graph_sequence_batch
from src.training.metrics import (
    compute_all_metrics, compute_per_variable_metrics, compute_scenario_metrics,
    bootstrap_metrics, paired_permutation_test,
    identify_geological_scenarios, compute_attention_geology_correlation,
)
from src.training.trainer import StandardizedHuberLoss, EarlyStopping, eval_epoch, eval_graph_sequence_epoch, train_graph_sequence_model, train_sequence_model, train_static_graph_model
from src.visualization.hotspot import aggregate_attention_to_surface
from src.visualization.style import (
    IJGIS_COLORS, IJGIS_CMAPS, PALETTE, add_panel_label, apply_publication_style,
    compute_unwrapped_coords, figure_size, save_figure, setup_unwrapped_surface_axes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

TARGET_NAMES = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
TARGET_LABELS = ["Advance Rate", "Torque", "Thrust", "Penetration", "Shield Pressure"]


# ── Helpers ────────────────────────────────────────────────────────

def build_sequence_samples(graph_snapshots, input_values, target_values, K, h):
    X_mon, y_list, graph_samples, target_chainages = [], [], [], []
    for i in range(len(graph_snapshots) - K - h + 1):
        graph_samples.append(graph_snapshots[i:i + K])
        X_mon.append(input_values[i:i + K])
        y_list.append(target_values[i + K + h - 1])
        target_chainages.append(graph_snapshots[i + K + h - 1].chainage)
    if not X_mon:
        return [], np.array([]), np.array([]), np.array([])
    return graph_samples, np.stack(X_mon), np.stack(y_list), np.asarray(target_chainages, dtype=np.float32)


def collect_graph_predictions(model, dataloader, device, tau):
    model.eval()
    preds = []
    with torch.no_grad():
        for graph_batch, mon_batch, _, _ in dataloader:
            pred, _ = model(graph_batch, mon_batch.to(device), tau=tau)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def collect_graph_attention(model, dataloader, device, tau):
    """收集模型原始注意力分数 s_ij (pre-softmax) 用于热点图可视化."""
    model.eval()
    all_attentions, all_edge_indices, all_n_tbm = [], [], []
    with torch.no_grad():
        for graph_batch, mon_batch, _, _ in dataloader:
            _, attentions = model(graph_batch, mon_batch.to(device), tau=tau, return_attention=True)
            for i, snap_seq in enumerate(graph_batch):
                last_snap = snap_seq[-1]
                if ("rock", "interact", "tbm") in last_snap.hetero_data.edge_types:
                    edge_idx = last_snap.hetero_data["rock", "interact", "tbm"].edge_index
                    n_tbm = last_snap.hetero_data["tbm"].x.shape[0]
                    # attentions[i] 是 s_ij (softmax 前的原始分数, 有绝对量纲区分度)
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
    stats_list = []
    for snap_seq in graph_seqs:
        last_snap = snap_seq[-1]
        rock_attrs = last_snap.rock_attrs.cpu().numpy()
        if len(rock_attrs) > 0:
            stats = np.concatenate([
                rock_attrs.mean(axis=0), rock_attrs.std(axis=0),
                rock_attrs.min(axis=0), rock_attrs.max(axis=0),
            ])
        else:
            stats = np.zeros(rock_attrs.shape[1] * 4 if len(rock_attrs.shape) > 1 else 8)
        stats_list.append(stats)
    return np.stack(stats_list)


# ── Figure 1: Framework Overview Pipeline ──────────────────────────

def gen_fig1_framework(output_dir: Path):
    """Generate the method overview pipeline figure (schematic diagram)."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figure_size("double", aspect=0.32))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    # Color definitions
    c_input = "#E8F4FD"
    c_process = "#D4E6F1"
    c_output = "#D5F5E3"
    c_arrow = "#333333"
    c_border = "#5B7C99"

    # Stage 1: Inputs
    boxes_input = [
        (0.3, 1.8, "TSP Voxel\nField\n$D_{geo}$", c_input),
        (0.3, 0.4, "TBM Surface\nModel\n$M_{TBM}$", c_input),
        (0.3, 0.0, "Monitoring\nRecords\n$u_t$", c_input),
    ]
    for x, y, txt, c in boxes_input:
        box = mpatches.FancyBboxPatch((x, y), 1.4, 1.1, boxstyle="round,pad=0.08",
                                       facecolor=c, edgecolor=c_border, linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 0.7, y + 0.55, txt, ha="center", va="center", fontsize=6.5)

    # Stage label
    ax.text(1.0, 3.0, "Stage 1: Spatial Entities", ha="center", fontsize=7,
            fontweight="bold", color=c_border)

    # Arrow 1→2
    ax.annotate("", xy=(2.2, 1.5), xytext=(1.8, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.2))

    # Stage 2: Graph Construction
    boxes_graph = [
        (2.4, 2.0, "Active Zone\nFiltering\n$\\Omega_t$", c_process),
        (2.4, 0.6, "Geometry-\nConstrained\nEdges $E^{rm}$", c_process),
    ]
    for x, y, txt, c in boxes_graph:
        box = mpatches.FancyBboxPatch((x, y), 1.4, 1.0, boxstyle="round,pad=0.08",
                                       facecolor=c, edgecolor=c_border, linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 0.7, y + 0.5, txt, ha="center", va="center", fontsize=6.5)

    # Graph snapshot symbol
    ax.text(3.1, 0.15, r"$G_t = (V^r \cup V^m,\ E^{rr} \cup E^{mm} \cup E^{rm})$",
            ha="center", fontsize=6, color="#333333", style="italic")
    ax.text(3.1, 3.0, "Stage 2: Graph Construction", ha="center", fontsize=7,
            fontweight="bold", color=c_border)

    # Arrow 2→3
    ax.annotate("", xy=(4.2, 1.5), xytext=(3.9, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.2))

    # Stage 3: Learning
    boxes_learn = [
        (4.4, 2.0, "Heterogeneous\nGNN Encoder\n$z_t, s_{ij}$", c_process),
        (4.4, 0.6, "GRU Temporal\nEncoder\n$\\hat{r}_{t+h}$", c_process),
    ]
    for x, y, txt, c in boxes_learn:
        box = mpatches.FancyBboxPatch((x, y), 1.4, 1.0, boxstyle="round,pad=0.08",
                                       facecolor=c, edgecolor=c_border, linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 0.7, y + 0.5, txt, ha="center", va="center", fontsize=6.5)

    # Supervision arrow
    ax.annotate("Response\nsupervision", xy=(5.1, 0.5), xytext=(5.1, -0.15),
                fontsize=5.5, ha="center", va="top", color="#C84C31",
                arrowprops=dict(arrowstyle="-|>", color="#C84C31", lw=0.8))
    ax.text(5.1, 3.0, "Stage 3: Interaction Learning", ha="center", fontsize=7,
            fontweight="bold", color=c_border)

    # Arrow 3→4
    ax.annotate("", xy=(6.2, 1.5), xytext=(5.9, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.2))

    # Stage 4: Interpretation
    boxes_interp = [
        (6.4, 2.0, "Surface\nHotspot Map\n$C_j$", c_output),
        (6.4, 0.6, "Chainage\nEvolution\nView", c_output),
    ]
    for x, y, txt, c in boxes_interp:
        box = mpatches.FancyBboxPatch((x, y), 1.4, 1.0, boxstyle="round,pad=0.08",
                                       facecolor=c, edgecolor=c_border, linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 0.7, y + 0.5, txt, ha="center", va="center", fontsize=6.5)

    ax.text(7.1, 3.0, "Stage 4: Interpretation", ha="center", fontsize=7,
            fontweight="bold", color=c_border)

    # Boundary note
    ax.text(9.5, 1.5,
            "Boundary:\n$s_{ij}$ is response-\nconsistent relevance,\nnot contact force",
            ha="center", va="center", fontsize=5.5, color="#888888",
            style="italic",
            bbox=dict(facecolor="#FFF8E1", edgecolor="#FFB300", linewidth=0.5,
                      boxstyle="round,pad=0.3"))

    fig.tight_layout()
    save_figure(fig, str(output_dir / "fig1_framework_pipeline.pdf"))
    logger.info("  Fig 1 saved.")


# ── Figure 2: Graph Construction 4-Panel ───────────────────────────

def gen_fig2_graph_construction(output_dir, snapshots, cfg, tbm_surface):
    """Generate the 4-panel graph construction figure."""
    from src.visualization.graph_viz import plot_graph_construction_figure

    tbm_cfg = cfg["tbm_geometry"]
    mid_idx = len(snapshots) // 2
    plot_graph_construction_figure(
        snapshot=snapshots[mid_idx],
        snapshots=snapshots,
        chainage_indices=[0, mid_idx, len(snapshots) - 1],
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        front_len=tbm_cfg["front_shield_len"],
        middle_len=tbm_cfg["middle_shield_len"],
        tail_len=tbm_cfg["tail_shield_len"],
        cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
        tau=cfg["graph"]["tau_edge"],
        save_path=str(output_dir / "fig2_graph_construction.pdf"),
    )
    logger.info("  Fig 2 saved.")


# ── Figure 3: Prediction Comparison ────────────────────────────────

def gen_fig3_prediction(output_dir, test_chainages, y_test, all_preds):
    """Generate per-variable prediction comparison figure (5 stacked panels)."""
    apply_publication_style()

    n_vars = len(TARGET_NAMES)
    fig, axes = plt.subplots(n_vars, 1, figsize=figure_size("double", aspect=0.22 * n_vars),
                             sharex=True)
    if n_vars == 1:
        axes = [axes]

    model_order = ["Full Model", "LSTM", "TSP-LSTM", "XGBoost", "Persistence"]
    model_styles = {
        "Full Model": {"color": IJGIS_COLORS["full_model"], "ls": "-", "lw": 1.5, "alpha": 0.95},
        "LSTM": {"color": IJGIS_COLORS["lstm"], "ls": "--", "lw": 1.0, "alpha": 0.85},
        "TSP-LSTM": {"color": IJGIS_COLORS["tsp_lstm"], "ls": "-.", "lw": 1.0, "alpha": 0.85},
        "XGBoost": {"color": IJGIS_COLORS["xgboost"], "ls": ":", "lw": 1.0, "alpha": 0.80},
        "Persistence": {"color": IJGIS_COLORS["persistence"], "ls": ":", "lw": 0.8, "alpha": 0.60},
    }

    for vi, (vname, vlabel) in enumerate(zip(TARGET_NAMES, TARGET_LABELS)):
        ax = axes[vi]
        truth = y_test[:, vi]

        # Observed
        ax.plot(test_chainages, truth, color=IJGIS_COLORS["truth"], linewidth=1.8,
                label="Observed", zorder=5)

        # Models
        for mi, mname in enumerate(model_order):
            if mname in all_preds:
                pred = all_preds[mname][:, vi]
                mae = np.mean(np.abs(pred - truth))
                sty = model_styles[mname]
                ax.plot(test_chainages, pred, color=sty["color"], linestyle=sty["ls"],
                        linewidth=sty["lw"], alpha=sty["alpha"],
                        label=f"{mname} (MAE={mae:.3f})", zorder=4 - mi)

        ax.set_ylabel(vlabel, fontsize=7)
        ax.grid(True, alpha=0.25)
        if vi == 0:
            ax.legend(fontsize=5.5, ncol=3, loc="upper right", framealpha=0.9)
            add_panel_label(ax, "a")
        else:
            add_panel_label(ax, chr(ord("a") + vi))

    axes[-1].set_xlabel("Chainage (m)")
    fig.tight_layout(h_pad=0.5)
    save_figure(fig, str(output_dir / "fig3_prediction_comparison.pdf"))
    logger.info("  Fig 3 saved.")


# ── Figure 4: Ablation Study ───────────────────────────────────────

def gen_fig4_ablation(output_dir, ablation_results, global_metrics):
    """Generate ablation study grouped bar chart."""
    apply_publication_style()

    # Combine full model + ablation variants
    all_variants = {"Full Framework": global_metrics.get("Full Model", {})}
    rename_map = {
        "No Monitoring": "w/o Monitoring",
        "Random Edges": "Randomized Edges",
        "No Geometry Prior": "w/o Geometric Prior",
        "No Geometric Constraints": "w/o Geometric Constraints",
    }
    for k, v in ablation_results.items():
        label = rename_map.get(k, k)
        all_variants[label] = v

    variant_names = list(all_variants.keys())
    metrics_to_show = ["MAE", "RMSE", r"$R^2$"]
    metric_keys = ["mae", "rmse", "r2"]

    n_variants = len(variant_names)
    n_metrics = len(metrics_to_show)
    x = np.arange(n_variants)
    width = 0.25

    fig, ax = plt.subplots(figsize=figure_size("double", aspect=0.45))

    colors = [IJGIS_COLORS["full_model"], IJGIS_COLORS["lstm"], "#E9C46A", "#C84C31"]

    for mi, (mkey, mlabel) in enumerate(zip(metric_keys, metrics_to_show)):
        values = [all_variants[v].get(mkey, 0) for v in variant_names]
        # Clip R2 for visualization
        if mkey == "r2":
            values = [max(v, -3.0) for v in values]
        offset = (mi - 1) * width
        bars = ax.bar(x + offset, values, width, label=mlabel, color=colors[mi], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Metric value")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.axhline(0, color="#666666", linewidth=0.5, linestyle=":")
    ax.grid(True, axis="y", alpha=0.25)
    add_panel_label(ax, "a")

    fig.tight_layout()
    save_figure(fig, str(output_dir / "fig4_ablation_study.pdf"))
    logger.info("  Fig 4 saved.")


# ── Figure 5: Hotspot Maps ─────────────────────────────────────────

def gen_fig5_hotspot(output_dir, test_chainages, test_graph_seqs,
                     attentions, edge_indices, n_tbm_list):
    """Generate multi-chainage hotspot maps on unwrapped TBM surface.

    Uses enlarged marker size and higher DPI for publication quality.
    """
    apply_publication_style()

    # Select 3 representative chainages (early, mid, late in test set)
    n_test = min(len(test_chainages), len(test_graph_seqs), len(attentions))
    if n_test == 0:
        logger.warning("  No test samples with attention, skipping Fig 5.")
        return
    indices = [0, n_test // 2, n_test - 1] if n_test >= 3 else list(range(n_test))

    fig, axes = plt.subplots(len(indices), 1,
                              figsize=figure_size("double", aspect=0.55 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for panel_i, sample_idx in enumerate(indices):
        ax = axes[panel_i]
        s_ij = attentions[sample_idx]  # raw attention scores (pre-softmax)
        eidx = edge_indices[sample_idx]
        n_tbm = n_tbm_list[sample_idx]

        if s_ij is None or eidx is None or n_tbm == 0:
            ax.text(0.5, 0.5, "No attention data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#999999")
            continue

        C_j, _ = aggregate_attention_to_surface(s_ij, eidx, n_tbm)
        snap = test_graph_seqs[sample_idx][-1]
        tbm_pos = snap.hetero_data["tbm"].x[:, :3].cpu().numpy()
        tbm_comp = snap.tbm_components.argmax(dim=1).cpu().numpy()

        # Shield nodes only
        shield_mask = tbm_comp >= 1
        pos = tbm_pos[shield_mask]
        C = C_j[shield_mask]

        if len(pos) == 0:
            continue

        x, theta = compute_unwrapped_coords(pos)

        # Component boundaries
        total_shield = 3.0 + 3.5 + 3.5  # front + mid + tail
        comp_bounds = [
            (0.0, "Cutterhead"),
            (3.0, "Front"),
            (6.5, "Middle"),
            (10.0, "Tail"),
        ]
        setup_unwrapped_surface_axes(ax, x_range=(x.min(), x.max()),
                                      component_boundaries=comp_bounds)

        # Enlarged marker size for visibility (s=28 for publication)
        vmax = np.percentile(C, 95) if len(C) > 0 else 1.0
        scatter = ax.scatter(x, theta, c=C, cmap=IJGIS_CMAPS["sequential"],
                             s=28, alpha=0.85, vmin=0, vmax=vmax,
                             edgecolors="none")

        chainage_val = float(test_chainages[sample_idx])
        ax.set_title(f"Chainage {chainage_val:.0f} m", fontsize=8, pad=3)
        add_panel_label(ax, chr(ord("a") + panel_i))

        # Colorbar for last panel only
        if panel_i == len(indices) - 1:
            cb = plt.colorbar(scatter, ax=axes, label=r"$C_j$ (surface relevance)",
                              shrink=0.8, pad=0.02)
            cb.ax.tick_params(labelsize=6)

    fig.tight_layout(h_pad=1.0)
    save_figure(fig, str(output_dir / "fig5_hotspot_maps.pdf"), dpi=300)
    logger.info("  Fig 5 saved (enlarged).")


# ── Figure 6: Chainage Evolution ───────────────────────────────────

def gen_fig6_chainage_evolution(output_dir, test_chainages, C_mean, y_test,
                                geo_chainages=None, geo_values=None):
    """Generate chainage evolution figure (relevance + geological + monitoring panels)."""
    apply_publication_style()

    n_vars = len(TARGET_NAMES)
    has_geo = geo_chainages is not None and geo_values is not None
    n_panels = 1 + (1 if has_geo else 0) + n_vars
    fig, axes = plt.subplots(n_panels, 1,
                              figsize=figure_size("double", aspect=0.18 * n_panels),
                              sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel a: Mean relevance
    ax0 = axes[0]
    ax0.fill_between(test_chainages, 0, C_mean, color=IJGIS_COLORS["accent"],
                     alpha=0.3)
    ax0.plot(test_chainages, C_mean, color=IJGIS_COLORS["accent"], linewidth=1.2)
    ax0.set_ylabel(r"Mean $C_j$", fontsize=7)
    ax0.grid(True, alpha=0.25)
    add_panel_label(ax0, "a")

    panel_idx = 1

    # Panel b: Geological context (TSP velocity)
    if has_geo:
        ax_geo = axes[panel_idx]
        ax_geo.plot(geo_chainages, geo_values, color="#33A02C", linewidth=1.0, alpha=0.8)
        ax_geo.set_ylabel("TSP velocity", fontsize=7)
        ax_geo.grid(True, alpha=0.25)
        # Mark low-velocity anomalies
        if len(geo_values) > 0:
            threshold = np.percentile(geo_values, 25)
            ax_geo.axhline(y=threshold, color="#E31A1C", linestyle="--", linewidth=0.7, alpha=0.6)
            ax_geo.fill_between(geo_chainages, geo_values, threshold,
                                where=geo_values < threshold,
                                color="#E31A1C", alpha=0.15)
        add_panel_label(ax_geo, chr(ord("a") + panel_idx))
        panel_idx += 1

    # Remaining panels: Monitoring variables
    for vi, (vname, vlabel) in enumerate(zip(TARGET_NAMES, TARGET_LABELS)):
        ax = axes[panel_idx + vi]
        ax.plot(test_chainages, y_test[:, vi], color=IJGIS_COLORS["full_model"],
                linewidth=0.9, alpha=0.85)
        ax.set_ylabel(vlabel, fontsize=7)
        ax.grid(True, alpha=0.25)
        add_panel_label(ax, chr(ord("a") + panel_idx + vi))

    axes[-1].set_xlabel("Chainage (m)")

    fig.tight_layout(h_pad=0.5)
    save_figure(fig, str(output_dir / "fig6_chainage_evolution.pdf"))
    logger.info("  Fig 6 saved.")


# ── Figure 7: Decision-Support Prototype System ────────────────────

def gen_fig7_decision_support(output_dir, test_chainages, y_test, full_preds,
                               C_mean_arr, geo_chainages, geo_values,
                               scenario_masks):
    """生成决策支持原型系统图 — GIS 导向的核心贡献图.

    展示方法如何支持隧道掘进决策:
      a) 预测响应 vs 实测 (含地质场景着色)
      b) 注意力预警信号 (C_mean 超阈值标记)
      c) 决策建议摘要 (场景标注 + 预警等级)
    """
    apply_publication_style()

    n_panels = 3
    fig, axes = plt.subplots(n_panels, 1,
                              figsize=figure_size("double", aspect=0.28 * n_panels),
                              sharex=True)

    # 场景颜色
    scenario_colors = {
        "Low-velocity zone": "#E31A1C",
        "Transition zone": "#FF7F00",
        "Normal zone": "#1B6CA8",
    }

    # 为每个测试样本分配场景标签
    sample_scenario = np.full(len(test_chainages), "Normal zone", dtype=object)
    for sname, mask in scenario_masks.items():
        sample_scenario[mask] = sname

    # Panel a: 预测 vs 实测 (含地质场景着色)
    ax0 = axes[0]
    ax0.plot(test_chainages, y_test[:, 0], color=IJGIS_COLORS["truth"],
             linewidth=1.5, label="Observed", zorder=3)
    ax0.plot(test_chainages, full_preds[:, 0], color=IJGIS_COLORS["full_model"],
             linewidth=1.2, linestyle="--", label="Predicted", zorder=4)

    # 地质场景背景着色 (逐连续段绘制, 避免覆盖中间非场景区域)
    for sname, color in scenario_colors.items():
        mask = sample_scenario == sname
        if not mask.any():
            continue
        # 找连续段
        indices = np.where(mask)[0]
        breaks = np.where(np.diff(indices) > 1)[0] + 1
        segments = np.split(indices, breaks)
        for seg in segments:
            if len(seg) > 0:
                ax0.axvspan(test_chainages[seg[0]], test_chainages[seg[-1]],
                            alpha=0.08, color=color, label=sname if seg is segments[0] else None,
                            zorder=1)

    ax0.set_ylabel("Advance Rate", fontsize=7)
    ax0.legend(fontsize=5.5, ncol=3, loc="upper right", framealpha=0.9)
    ax0.grid(True, alpha=0.2)
    add_panel_label(ax0, "a")

    # Panel b: 注意力预警信号
    ax1 = axes[1]
    ax1.fill_between(test_chainages, 0, C_mean_arr, color=IJGIS_COLORS["accent"],
                     alpha=0.3)
    ax1.plot(test_chainages, C_mean_arr, color=IJGIS_COLORS["accent"],
             linewidth=1.2, label=r"Mean $C_j$")

    # 预警阈值 (P75)
    if len(C_mean_arr) > 0:
        alert_threshold = np.percentile(C_mean_arr, 75)
        ax1.axhline(y=alert_threshold, color="#E31A1C", linestyle="--",
                     linewidth=0.8, alpha=0.7, label=f"Alert (P75={alert_threshold:.2f})")
        # 标记超阈值区域
        alert_mask = C_mean_arr >= alert_threshold
        if alert_mask.any():
            ax1.fill_between(test_chainages, alert_threshold, C_mean_arr,
                             where=alert_mask, color="#E31A1C", alpha=0.2)

    # TSP 速度叠加 (右轴)
    ax1r = ax1.twinx()
    test_vp = np.interp(test_chainages, geo_chainages, geo_values)
    ax1r.plot(test_chainages, test_vp, color="#33A02C", linewidth=0.8,
              alpha=0.6, label="TSP Vp")
    ax1r.set_ylabel("TSP Vp (m/s)", fontsize=6, color="#33A02C")
    ax1r.tick_params(axis="y", labelsize=6, colors="#33A02C")

    ax1.set_ylabel(r"Mean $C_j$", fontsize=7)
    ax1.legend(fontsize=5.5, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    add_panel_label(ax1, "b")

    # Panel c: 决策建议摘要
    ax2 = axes[2]
    # 综合预警指数: 注意力归一化 + 预测误差归一化
    pred_err = np.abs(full_preds[:, 0] - y_test[:, 0])
    if C_mean_arr.max() > C_mean_arr.min():
        C_norm = (C_mean_arr - C_mean_arr.min()) / (C_mean_arr.max() - C_mean_arr.min() + 1e-8)
    else:
        C_norm = np.zeros_like(C_mean_arr)
    if pred_err.max() > pred_err.min():
        err_norm = (pred_err - pred_err.min()) / (pred_err.max() - pred_err.min() + 1e-8)
    else:
        err_norm = np.zeros_like(pred_err)

    risk_index = 0.5 * C_norm + 0.5 * err_norm

    # 三级预警
    risk_colors = np.where(risk_index < 0.33, "#2A9D8F",
                  np.where(risk_index < 0.66, "#E9C46A", "#E31A1C"))

    bar_width = float(np.diff(test_chainages).min()) if len(test_chainages) > 1 else 0.5
    ax2.bar(test_chainages, risk_index, width=bar_width,
            color=risk_colors, alpha=0.8)

    # 场景标注 (逐连续段)
    for sname, color in scenario_colors.items():
        mask = sample_scenario == sname
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        breaks = np.where(np.diff(indices) > 1)[0] + 1
        segments = np.split(indices, breaks)
        for seg in segments:
            if len(seg) > 0:
                ax2.axvspan(test_chainages[seg[0]], test_chainages[seg[-1]],
                            alpha=0.06, color=color, zorder=0)

    ax2.set_ylabel("Risk index", fontsize=7)
    ax2.set_ylim(0, 1.1)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2A9D8F", alpha=0.8, label="Low risk"),
        Patch(facecolor="#E9C46A", alpha=0.8, label="Medium risk"),
        Patch(facecolor="#E31A1C", alpha=0.8, label="High risk"),
    ]
    ax2.legend(handles=legend_elements, fontsize=5.5, loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    add_panel_label(ax2, "c")

    axes[-1].set_xlabel("Chainage (m)")
    fig.tight_layout(h_pad=0.8)
    save_figure(fig, str(output_dir / "fig7_decision_support.pdf"))
    logger.info("  Fig 7 saved.")


# ── Figure 8: Geological Scenario Validation ───────────────────────

def gen_fig8_scenario_validation(output_dir, y_test, all_preds, scenario_masks):
    """地质场景验证图 — GIS 导向的核心评估.

    不再只看全局指标, 而是展示模型在不同地质场景下的空间适应性:
      a) 分场景 MAE 对比 (分组柱状图)
      b) 分场景 R² 对比
      c) 场景内预测 vs 实测散点图
    """
    apply_publication_style()

    scenario_names = list(scenario_masks.keys())
    n_scenarios = len(scenario_names)
    if n_scenarios == 0:
        logger.warning("  No geological scenarios identified, skipping Fig 8.")
        return

    model_order = ["Full Model", "TSP-LSTM", "LSTM", "SpatialAgg-LSTM"]
    model_colors = {
        "Full Model": IJGIS_COLORS["full_model"],
        "TSP-LSTM": IJGIS_COLORS["tsp_lstm"],
        "LSTM": IJGIS_COLORS["lstm"],
        "SpatialAgg-LSTM": "#9A9A9A",
    }

    # Panel a: 分场景 MAE 对比
    fig, axes = plt.subplots(1, 3, figsize=figure_size("double", aspect=0.55))

    # 收集分场景指标
    scenario_model_mae = {}
    scenario_model_r2 = {}
    for mname, preds in all_preds.items():
        if mname not in model_order:
            continue
        scenario_model_mae[mname] = {}
        scenario_model_r2[mname] = {}
        for sname, mask in scenario_masks.items():
            if mask.sum() < 2:
                continue
            m = compute_all_metrics(y_test[mask], preds[mask])
            scenario_model_mae[mname][sname] = m["mae"]
            scenario_model_r2[mname][sname] = max(m["r2"], -3.0)

    # Panel a: MAE
    ax0 = axes[0]
    x = np.arange(n_scenarios)
    width = 0.18
    for mi, mname in enumerate(model_order):
        if mname not in scenario_model_mae:
            continue
        vals = [scenario_model_mae[mname].get(s, 0) for s in scenario_names]
        ax0.bar(x + mi * width, vals, width, label=mname,
                color=model_colors[mname], alpha=0.85)
    ax0.set_xticks(x + width * (len(model_order) - 1) / 2)
    ax0.set_xticklabels(scenario_names, fontsize=6, rotation=15, ha="right")
    ax0.set_ylabel("MAE")
    ax0.legend(fontsize=5.5, loc="upper right")
    ax0.grid(True, axis="y", alpha=0.2)
    add_panel_label(ax0, "a")

    # Panel b: R²
    ax1 = axes[1]
    for mi, mname in enumerate(model_order):
        if mname not in scenario_model_r2:
            continue
        vals = [scenario_model_r2[mname].get(s, 0) for s in scenario_names]
        ax1.bar(x + mi * width, vals, width, label=mname,
                color=model_colors[mname], alpha=0.85)
    ax1.set_xticks(x + width * (len(model_order) - 1) / 2)
    ax1.set_xticklabels(scenario_names, fontsize=6, rotation=15, ha="right")
    ax1.set_ylabel(r"$R^2$")
    ax1.axhline(0, color="#666666", linewidth=0.5, linestyle=":")
    ax1.grid(True, axis="y", alpha=0.2)
    add_panel_label(ax1, "b")

    # Panel c: 散点图 (Full Model only, 按场景着色)
    ax2 = axes[2]
    scenario_scatter_colors = {
        "Low-velocity zone": "#E31A1C",
        "Transition zone": "#FF7F00",
        "Normal zone": "#1B6CA8",
    }
    full_preds_for_scatter = all_preds.get("Full Model", np.zeros_like(y_test))
    for sname, mask in scenario_masks.items():
        if mask.sum() < 1:
            continue
        color = scenario_scatter_colors.get(sname, "#999999")
        ax2.scatter(y_test[mask, 0], full_preds_for_scatter[mask, 0],
                    c=color, s=12, alpha=0.7, label=sname, edgecolors="white", linewidths=0.3)
    # 45度线
    vmin = min(y_test[:, 0].min(), full_preds_for_scatter[:, 0].min())
    vmax = max(y_test[:, 0].max(), full_preds_for_scatter[:, 0].max())
    ax2.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("Observed")
    ax2.set_ylabel("Predicted")
    ax2.legend(fontsize=5.5, loc="upper left")
    ax2.grid(True, alpha=0.2)
    add_panel_label(ax2, "c")

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, str(output_dir / "fig8_scenario_validation.pdf"))
    logger.info("  Fig 8 saved.")


# ── Main Pipeline ──────────────────────────────────────────────────

def main(config_path: str = "config/default.yaml"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    runtime_cfg = cfg.get("runtime", {})

    SEED = cfg.get("seed", 42)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}, seed: {SEED}")

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
    logger.info("Loading data...")
    tsp_df = load_tsp(cfg["data"]["tsp_path"])
    mon_df = load_monitoring(cfg["data"]["monitoring_path"])
    mon_df = aggregate_by_chainage(mon_df, cfg["excavation"]["step_length"])

    K = cfg["model"]["history_window"]
    h = cfg["model"]["predict_horizon"]
    n_total = len(mon_df) - K - h + 1
    train_sl, val_sl, test_sl = mileage_split(n_total)
    train_fit_stop = train_sl.stop

    # 仅用训练集里程范围内的 TSP 体素计算统计量，避免数据泄漏
    tsp_df = normalize_coords(tsp_df)
    train_chainage_max = mon_df["chainage"].iloc[train_sl.stop - 1]
    tsp_x_max = train_chainage_max - mon_df["chainage"].min()
    train_tsp_mask = tsp_df["X_local"].to_numpy() <= tsp_x_max
    train_tsp_raw = tsp_df.loc[train_tsp_mask, TSP_ATTR_COLS].to_numpy(dtype=np.float32)
    attr_mean = train_tsp_raw.mean(axis=0, keepdims=True)
    attr_std = train_tsp_raw.std(axis=0, keepdims=True) + 1e-8
    rock_coords = get_rock_coords(tsp_df)
    rock_attrs = get_rock_attrs(tsp_df, mean=attr_mean, std=attr_std)

    mon_df, scaler = standardize_monitoring(mon_df, fit_df=mon_df.iloc[:train_fit_stop])

    tbm_cfg = cfg["tbm_geometry"]
    tbm_surface = build_tbm_surface(
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        front_len=tbm_cfg["front_shield_len"],
        middle_len=tbm_cfg["middle_shield_len"],
        tail_len=tbm_cfg["tail_shield_len"],
        resolution=tbm_cfg["surface_resolution"],
    )

    tau = cfg["graph"]["tau_edge"]
    eta_min = cfg["graph"]["normal_threshold"]

    steps = build_excavation_steps(mon_df, step_length=cfg["excavation"]["step_length"], K=K, h=h)
    logger.info(f"Building graph sequence ({len(steps)} steps)...")
    snapshots = build_graph_sequence(
        steps, rock_coords, rock_attrs, tbm_surface,
        cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
        cutterhead_radius=tbm_cfg["cutterhead_radius"],
        shield_radius=tbm_cfg["shield_radius"],
        tau_zone=cfg["graph"].get("tau_zone", 5.0),
        tau_edge=tau, eta_min=eta_min,
        device=device, verbose=True,
    )

    mon_features = mon_df[["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]].to_numpy(dtype=np.float32)
    mon_targets = mon_df[["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]].to_numpy(dtype=np.float32)

    graph_seqs, X_mon, y, sample_chainages = build_sequence_samples(snapshots, mon_features, mon_targets, K, h)

    X_mon_t = torch.tensor(X_mon, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    train_ds = TensorDataset(X_mon_t[train_sl], y_t[train_sl])
    val_ds = TensorDataset(X_mon_t[val_sl], y_t[val_sl])
    test_ds = TensorDataset(X_mon_t[test_sl], y_t[test_sl])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])

    train_gds = GraphSequenceDataset(graph_seqs[train_sl], X_mon[train_sl], y[train_sl], sample_chainages[train_sl])
    val_gds = GraphSequenceDataset(graph_seqs[val_sl], X_mon[val_sl], y[val_sl], sample_chainages[val_sl])
    test_gds = GraphSequenceDataset(graph_seqs[test_sl], X_mon[test_sl], y[test_sl], sample_chainages[test_sl])
    graph_train_loader = DataLoader(train_gds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_graph_sequence_batch)
    graph_val_loader = DataLoader(val_gds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_graph_sequence_batch)
    graph_test_loader = DataLoader(test_gds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_graph_sequence_batch)

    tsp_stats = compute_tsp_stats_per_sample(graph_seqs, K)

    rock_in_dim = rock_coords.shape[1] + rock_attrs.shape[1] + 1
    tbm_in_dim = 3 + 3 + 4
    edge_in_dim = 1 + 1 + 3 + rock_attrs.shape[1] + 4

    train_kwargs = dict(
        epochs=cfg["training"]["epochs"], lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"], patience=cfg["training"]["patience"],
        huber_delta=cfg["training"]["huber_delta"], device=device,
    )

    # ── Train all models ───────────────────────────────────────────
    logger.info("Training models...")

    # 1. Persistence
    target_feature_idx = [0, 2, 3, 4, 5]
    persist_preds = X_mon[test_sl, -1, target_feature_idx]
    persist_metrics = compute_all_metrics(y[test_sl], persist_preds)

    # 2. XGBoost
    xgb = TSPXGBoost(n_estimators=runtime_cfg.get("xgb_n_estimators", 200),
                      max_depth=runtime_cfg.get("xgb_max_depth", 6))
    xgb.fit(X_mon[train_sl], tsp_stats[train_sl], y[train_sl])
    xgb_preds = xgb.predict(X_mon[test_sl], tsp_stats[test_sl])
    xgb_metrics = compute_all_metrics(y[test_sl], xgb_preds)

    # 3. LSTM
    lstm = LSTMBaseline(input_dim=6, hidden_dim=128, num_layers=2, output_dim=5, dropout=0.1).to(device)
    lstm, _ = train_sequence_model(lstm, train_loader, val_loader, checkpoint_dir=output_dir, **train_kwargs)
    lstm.eval()
    with torch.no_grad():
        lstm_preds = lstm(X_mon_t[test_sl].to(device)).cpu().numpy()
    lstm_metrics = compute_all_metrics(y[test_sl], lstm_preds)

    # 4. TSP-LSTM
    tsp_stat_dim = tsp_stats.shape[1]
    tsp_lstm = TSPLSTM(input_dim=6, tsp_stat_dim=tsp_stat_dim, hidden_dim=128,
                       num_layers=2, output_dim=5, dropout=0.1).to(device)
    tsp_train_ds = TensorDataset(X_mon_t[train_sl], torch.tensor(tsp_stats[train_sl], dtype=torch.float32), y_t[train_sl])
    tsp_val_ds = TensorDataset(X_mon_t[val_sl], torch.tensor(tsp_stats[val_sl], dtype=torch.float32), y_t[val_sl])
    tsp_train_loader = DataLoader(tsp_train_ds, batch_size=cfg["training"]["batch_size"])
    tsp_val_loader = DataLoader(tsp_val_ds, batch_size=cfg["training"]["batch_size"])
    loss_fn = StandardizedHuberLoss(n_targets=5, delta=train_kwargs["huber_delta"])
    optimizer = torch.optim.AdamW(tsp_lstm.parameters(), lr=train_kwargs["lr"], weight_decay=train_kwargs["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
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
            break
    if best_state:
        tsp_lstm.load_state_dict(best_state)
    tsp_lstm.eval()
    with torch.no_grad():
        tsp_lstm_preds = tsp_lstm(X_mon_t[test_sl].to(device),
                                   torch.tensor(tsp_stats[test_sl], dtype=torch.float32).to(device)).cpu().numpy()
    tsp_lstm_metrics = compute_all_metrics(y[test_sl], tsp_lstm_preds)

    # 4.5. SpatialAggLSTM — 简单空间聚合基线 (验证图结构必要性)
    logger.info("Computing spatial aggregation features for SpatialAggLSTM baseline...")
    spatial_agg_feats = compute_spatial_agg_features(graph_seqs)
    spatial_dim = spatial_agg_feats.shape[-1]
    spatial_agg_lstm = SpatialAggLSTM(
        input_dim=6, spatial_dim=spatial_dim, hidden_dim=128,
        num_layers=2, output_dim=5, dropout=0.1,
    ).to(device)

    spa_train_ds = TensorDataset(
        X_mon_t[train_sl],
        torch.tensor(spatial_agg_feats[train_sl], dtype=torch.float32),
        y_t[train_sl],
    )
    spa_val_ds = TensorDataset(
        X_mon_t[val_sl],
        torch.tensor(spatial_agg_feats[val_sl], dtype=torch.float32),
        y_t[val_sl],
    )
    spa_train_loader = DataLoader(spa_train_ds, batch_size=cfg["training"]["batch_size"])
    spa_val_loader = DataLoader(spa_val_ds, batch_size=cfg["training"]["batch_size"])

    spa_optimizer = torch.optim.AdamW(spatial_agg_lstm.parameters(),
                                       lr=train_kwargs["lr"],
                                       weight_decay=train_kwargs["weight_decay"])
    spa_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(spa_optimizer, mode="min", factor=0.5, patience=10)
    spa_es = EarlyStopping(patience=train_kwargs["patience"])
    spa_best_state = None
    for epoch in range(1, train_kwargs["epochs"] + 1):
        spatial_agg_lstm.train()
        for x_seq, spa_feat, yt in spa_train_loader:
            x_seq, spa_feat, yt = x_seq.to(device), spa_feat.to(device), yt.to(device)
            spa_optimizer.zero_grad()
            pred = spatial_agg_lstm(x_seq, spa_feat)
            loss = loss_fn(pred, yt)
            loss.backward()
            spa_optimizer.step()
        spatial_agg_lstm.eval()
        spa_val_loss = 0.0
        with torch.no_grad():
            for x_seq, spa_feat, yt in spa_val_loader:
                pred = spatial_agg_lstm(x_seq.to(device), spa_feat.to(device))
                spa_val_loss += loss_fn(pred, yt.to(device)).item()
        spa_val_loss /= len(spa_val_loader)
        spa_scheduler.step(spa_val_loss)
        if spa_es.update(spa_val_loss):
            spa_best_state = {k: v.cpu().clone() for k, v in spatial_agg_lstm.state_dict().items()}
        if spa_es.should_stop:
            break
    if spa_best_state:
        spatial_agg_lstm.load_state_dict(spa_best_state)
    spatial_agg_lstm.eval()
    with torch.no_grad():
        spa_preds = spatial_agg_lstm(
            X_mon_t[test_sl].to(device),
            torch.tensor(spatial_agg_feats[test_sl], dtype=torch.float32).to(device),
        ).cpu().numpy()
    spa_metrics = compute_all_metrics(y[test_sl], spa_preds)
    logger.info(f"  SpatialAggLSTM — MAE: {spa_metrics['mae']:.4f}, R²: {spa_metrics['r2']:.4f}")

    # 5. Full Model
    full_model = GraphSequenceModel(
        rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
        monitoring_dim=6, output_dim=5,
        rock_hidden=cfg["model"]["rock_node_dim"], tbm_hidden=cfg["model"]["tbm_node_dim"],
        edge_hidden=cfg["model"]["edge_dim"], gnn_layers=cfg["model"]["gnn_layers"],
        gru_hidden=cfg["model"]["gru_hidden_dim"], gru_layers=cfg["model"]["gru_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    full_model, _ = train_graph_sequence_model(full_model, graph_train_loader, graph_val_loader,
                                                tau=tau, checkpoint_dir=output_dir, **train_kwargs)
    full_preds = collect_graph_predictions(full_model, graph_test_loader, device, tau)
    full_metrics = compute_all_metrics(y[test_sl], full_preds)

    # ── Ablation models ────────────────────────────────────────────
    ablation_results = {}
    ablation_preds = {}

    if runtime_cfg.get("run_ablations", True):
        # A1: No monitoring
        graph_only = DynamicGraphOnly(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            monitoring_dim=6, output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"], tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"], gnn_layers=cfg["model"]["gnn_layers"],
            gru_hidden=cfg["model"]["gru_hidden_dim"], gru_layers=cfg["model"]["gru_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        graph_only, _ = train_graph_sequence_model(graph_only, graph_train_loader, graph_val_loader,
                                                     tau=tau, checkpoint_dir=output_dir,
                                                     checkpoint_name="ablation_no_mon.pt", **train_kwargs)
        a1_preds = collect_graph_predictions(graph_only, graph_test_loader, device, tau)
        ablation_results["No Monitoring"] = compute_all_metrics(y[test_sl], a1_preds)
        ablation_preds["No Monitoring"] = a1_preds

        # A2: Random edges
        rand_model = RandomEdgesGraphSeq(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            monitoring_dim=6, output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"], tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"], gnn_layers=cfg["model"]["gnn_layers"],
            gru_hidden=cfg["model"]["gru_hidden_dim"], gru_layers=cfg["model"]["gru_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        rand_model, _ = train_graph_sequence_model(rand_model, graph_train_loader, graph_val_loader,
                                                     tau=tau, checkpoint_dir=output_dir,
                                                     checkpoint_name="ablation_random_edges.pt", **train_kwargs)
        a2_preds = collect_graph_predictions(rand_model, graph_test_loader, device, tau)
        ablation_results["Random Edges"] = compute_all_metrics(y[test_sl], a2_preds)
        ablation_preds["Random Edges"] = a2_preds

        # A3: No geometric prior
        no_prior = NoGeometryPrior(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            monitoring_dim=6, output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"], tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"], gnn_layers=cfg["model"]["gnn_layers"],
            gru_hidden=cfg["model"]["gru_hidden_dim"], gru_layers=cfg["model"]["gru_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        no_prior, _ = train_graph_sequence_model(no_prior, graph_train_loader, graph_val_loader,
                                                   tau=tau, checkpoint_dir=output_dir,
                                                   checkpoint_name="ablation_no_prior.pt", **train_kwargs)
        a3_preds = collect_graph_predictions(no_prior, graph_test_loader, device, tau)
        ablation_results["No Geometry Prior"] = compute_all_metrics(y[test_sl], a3_preds)
        ablation_preds["No Geometry Prior"] = a3_preds

        # A4: No geometric constraints (relaxed graph construction) — optional, slow
        if runtime_cfg.get("run_relaxed_ablation", False):
            logger.info("Building relaxed graph sequence for NoGeometricConstraints ablation...")
            relaxed_snapshots = build_graph_sequence(
                steps, rock_coords, rock_attrs, tbm_surface,
                cutterhead_look_ahead=cfg["excavation"].get("cutterhead_look_ahead", 5.0),
                cutterhead_radius=tbm_cfg["cutterhead_radius"],
                shield_radius=tbm_cfg["shield_radius"],
                tau_zone=10.0,  # 宽松径向范围
                tau_edge=10.0,  # 宽松候选边距离阈值
                eta_min=0.0,    # 无法向一致性约束
                device=device, verbose=False,
            )
            relaxed_graph_seqs, _, _, _ = build_sequence_samples(
                relaxed_snapshots, mon_features, mon_targets, K, h
            )
            relaxed_train_gds = GraphSequenceDataset(
                relaxed_graph_seqs[train_sl], X_mon[train_sl], y[train_sl], sample_chainages[train_sl]
            )
            relaxed_val_gds = GraphSequenceDataset(
                relaxed_graph_seqs[val_sl], X_mon[val_sl], y[val_sl], sample_chainages[val_sl]
            )
            relaxed_test_gds = GraphSequenceDataset(
                relaxed_graph_seqs[test_sl], X_mon[test_sl], y[test_sl], sample_chainages[test_sl]
            )
            relaxed_train_loader = DataLoader(relaxed_train_gds, batch_size=cfg["training"]["batch_size"],
                                              shuffle=False, collate_fn=collate_graph_sequence_batch)
            relaxed_val_loader = DataLoader(relaxed_val_gds, batch_size=cfg["training"]["batch_size"],
                                            shuffle=False, collate_fn=collate_graph_sequence_batch)
            relaxed_test_loader = DataLoader(relaxed_test_gds, batch_size=cfg["training"]["batch_size"],
                                             shuffle=False, collate_fn=collate_graph_sequence_batch)

            no_constraints = NoGeometricConstraints(
                rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
                monitoring_dim=6, output_dim=5,
                rock_hidden=cfg["model"]["rock_node_dim"], tbm_hidden=cfg["model"]["tbm_node_dim"],
                edge_hidden=cfg["model"]["edge_dim"], gnn_layers=cfg["model"]["gnn_layers"],
                gru_hidden=cfg["model"]["gru_hidden_dim"], gru_layers=cfg["model"]["gru_layers"],
                dropout=cfg["model"]["dropout"],
            ).to(device)
            no_constraints, _ = train_graph_sequence_model(
                no_constraints, relaxed_train_loader, relaxed_val_loader,
                tau=10.0, checkpoint_dir=output_dir,
                checkpoint_name="ablation_no_constraints.pt", **train_kwargs
            )
            a4_preds = collect_graph_predictions(no_constraints, relaxed_test_loader, device, tau=10.0)
            ablation_results["No Geometric Constraints"] = compute_all_metrics(y[test_sl], a4_preds)
            ablation_preds["No Geometric Constraints"] = a4_preds
        else:
            logger.info("Skipping NoGeometricConstraints ablation (run_relaxed_ablation=false)")

        # A5: Static graph (single G_t, no sequence)
        logger.info("Training StaticGraph ablation (single G_t, no GRU)...")
        static_model = StaticGraphModel(
            rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
            output_dim=5,
            rock_hidden=cfg["model"]["rock_node_dim"], tbm_hidden=cfg["model"]["tbm_node_dim"],
            edge_hidden=cfg["model"]["edge_dim"], num_layers=cfg["model"]["gnn_layers"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        static_model, _ = train_static_graph_model(
            static_model, graph_train_loader, graph_val_loader,
            tau=tau, checkpoint_dir=output_dir,
            checkpoint_name="ablation_static_graph.pt",
            epochs=train_kwargs.get("epochs", 200),
            lr=train_kwargs.get("lr", 0.001),
            weight_decay=train_kwargs.get("weight_decay", 0.01),
            patience=train_kwargs.get("patience", 30),
            huber_delta=train_kwargs.get("huber_delta", 1.0),
            device=device,
        )
        # 收集静态图预测
        static_model.eval()
        static_preds_list = []
        with torch.no_grad():
            for batch in graph_test_loader:
                graph_seqs, _, y_batch = batch[:3]
                # 遍历 batch 中每个样本, 取其序列的最后一个图快照
                for sample_seqs in graph_seqs:
                    last_snap = sample_seqs[-1]
                    rock_attrs_t = last_snap.rock_attrs.to(device)
                    tbm_comp_t = last_snap.tbm_components.to(device)
                    pred = static_model(last_snap.hetero_data, rock_attrs_t, tbm_comp_t, tau=tau)
                    static_preds_list.append(pred.unsqueeze(0).cpu().numpy())  # (1, 5)
        a5_preds = np.concatenate(static_preds_list, axis=0)  # (N, 5)
        ablation_results["Static Graph"] = compute_all_metrics(y[test_sl], a5_preds)
        ablation_preds["Static Graph"] = a5_preds

    # ── Attention extraction ────────────────────────────────────────
    logger.info("Extracting attention weights...")
    attentions, edge_indices, n_tbm_list = collect_graph_attention(
        full_model, graph_test_loader, device, tau
    )

    C_j_samples = []
    for s_ij, eidx, n_tbm in zip(attentions, edge_indices, n_tbm_list):
        if s_ij is not None and eidx is not None:
            C_j, _ = aggregate_attention_to_surface(s_ij, eidx, n_tbm)
            C_j_samples.append(C_j)
        else:
            C_j_samples.append(None)

    C_mean_list = []
    for C_j in C_j_samples:
        if C_j is not None and len(C_j) > 0:
            C_mean_list.append(float(C_j.mean()))
        else:
            C_mean_list.append(0.0)
    C_mean_arr = np.array(C_mean_list)

    # ── Prepare geological context for Fig 6 ───────────────────────
    # 按 X_local 聚合 TSP P-wave velocity，用于地质标注
    tsp_x = tsp_df["X_local"].to_numpy()
    tsp_vp = tsp_df["Vp"].to_numpy()
    # 按 0.5m bin 聚合
    geo_bin_size = 0.5
    geo_x_bins = np.arange(tsp_x.min(), tsp_x.max() + geo_bin_size, geo_bin_size)
    geo_chainages = np.array([(geo_x_bins[i] + geo_x_bins[i + 1]) / 2
                               for i in range(len(geo_x_bins) - 1)])
    geo_values = np.array([tsp_vp[(tsp_x >= geo_x_bins[i]) & (tsp_x < geo_x_bins[i + 1])].mean()
                            if np.any((tsp_x >= geo_x_bins[i]) & (tsp_x < geo_x_bins[i + 1]))
                            else np.nan
                            for i in range(len(geo_x_bins) - 1)])
    # 去除 NaN
    valid = ~np.isnan(geo_values)
    geo_chainages = geo_chainages[valid]
    geo_values = geo_values[valid].astype(np.float32)

    # ── Generate all figures ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Generating all publication figures...")

    test_chainages = sample_chainages[test_sl]
    y_test = y[test_sl]
    test_graph_seqs = graph_seqs[test_sl]

    logger.info(f"  Test set: {len(test_chainages)} chainages, {len(test_graph_seqs)} graph seqs, {len(attentions)} attentions")

    all_preds = {
        "Full Model": full_preds,
        "LSTM": lstm_preds,
        "TSP-LSTM": tsp_lstm_preds,
        "SpatialAgg-LSTM": spa_preds,
        "XGBoost": xgb_preds,
        "Persistence": persist_preds,
    }
    global_metrics = {
        "Persistence": persist_metrics,
        "XGBoost": xgb_metrics,
        "LSTM": lstm_metrics,
        "TSP-LSTM": tsp_lstm_metrics,
        "SpatialAgg-LSTM": spa_metrics,
        "Full Model": full_metrics,
    }

    # ── 报告训练后 β 值 ──────────────────────────────────────────
    beta_val = full_model.gnn.beta.item()
    logger.info(f"  Learned geometry prior β = {beta_val:.4f}")

    # ── 地质场景识别 ──────────────────────────────────────────────
    logger.info("Identifying geological scenarios...")
    scenario_masks = identify_geological_scenarios(
        geo_chainages, geo_values, test_chainages,
        low_velocity_percentile=25.0, transition_half_width=5.0,
    )
    for sname, mask in scenario_masks.items():
        logger.info(f"  Scenario '{sname}': {mask.sum()} test samples")

    # ── 注意力-地质相关性分析 ─────────────────────────────────────
    attn_geo_corr = compute_attention_geology_correlation(
        C_mean_arr, test_chainages, geo_chainages, geo_values,
    )
    logger.info(f"  Attention-geology correlation: "
                f"Pearson r={attn_geo_corr['pearson_r']:.3f} (p={attn_geo_corr['pearson_p']:.4f}), "
                f"Spearman r={attn_geo_corr['spearman_r']:.3f} (p={attn_geo_corr['spearman_p']:.4f})")

    # ── 空间一致性指标 ─────────────────────────────────────────────
    from src.training.metrics import compute_morans_i, compute_component_cv
    logger.info("Computing spatial consistency indicators...")

    spatial_consistency = {}
    # Full model spatial consistency
    if len(C_j_samples) > 0 and C_j_samples[0] is not None:
        # Aggregate all test C_j into a single array with TBM positions
        all_C_j = np.concatenate(C_j_samples)
        # Get TBM positions from the last test snapshot
        test_snap = test_graph_seqs[len(test_graph_seqs) // 2][-1]  # mid-test snapshot
        tbm_pos = test_snap.tbm_positions.numpy() if hasattr(test_snap.tbm_positions, 'numpy') else test_snap.tbm_positions
        tbm_comp = test_snap.tbm_components.numpy() if hasattr(test_snap.tbm_components, 'numpy') else test_snap.tbm_components

        # Moran's I (mean across chainages)
        moran_values = []
        for C_j in C_j_samples:
            if C_j is not None and len(C_j) == len(tbm_pos):
                mi = compute_morans_i(C_j, tbm_pos)
                moran_values.append(mi)
        morans_i_mean = float(np.mean(moran_values)) if moran_values else 0.0

        # Component CV
        cv_values = []
        for C_j in C_j_samples:
            if C_j is not None and len(C_j) == len(tbm_comp):
                cv = compute_component_cv(C_j, tbm_comp)
                cv_values.append(cv)
        comp_cv_mean = float(np.mean(cv_values)) if cv_values else 0.0

        spatial_consistency["Full Model"] = {
            "attention_geology_r": attn_geo_corr["pearson_r"],
            "morans_i": morans_i_mean,
            "component_cv": comp_cv_mean,
        }
        logger.info(f"  Full Model: Moran's I={morans_i_mean:.3f}, CV={comp_cv_mean:.3f}")

    # Ablation spatial consistency (for No Geometry Prior and Random Edges)
    if runtime_cfg.get("run_ablations", True):
        for ablation_name, ablation_model, ablation_ckpt in [
            ("No Geometry Prior", NoGeometryPrior, "ablation_no_prior.pt"),
            ("Random Edges", RandomEdgesGraphSeq, "ablation_random_edges.pt"),
        ]:
            try:
                ab_attentions, ab_edge_indices, ab_n_tbm = collect_graph_attention(
                    ablation_model(
                        rock_in_dim=rock_in_dim, tbm_in_dim=tbm_in_dim, edge_in_dim=edge_in_dim,
                        monitoring_dim=6, output_dim=5,
                        rock_hidden=cfg["model"]["rock_node_dim"], tbm_hidden=cfg["model"]["tbm_node_dim"],
                        edge_hidden=cfg["model"]["edge_dim"], gnn_layers=cfg["model"]["gnn_layers"],
                        gru_hidden=cfg["model"]["gru_hidden_dim"], gru_layers=cfg["model"]["gru_layers"],
                        dropout=cfg["model"]["dropout"],
                    ).to(device), graph_test_loader, device, tau
                )
                # This won't work because we need the trained model, not a new one
                # Skip for now — will compute from saved model
            except Exception:
                pass

        # Compute spatial consistency for ablation variants using saved models
        # We need to reload the trained ablation models and extract attention
        # For now, compute from the trained models we already have
        ablation_spatial = {}

        # No Geometry Prior
        if "No Geometry Prior" in ablation_results:
            try:
                no_prior_attn, no_prior_eidx, no_prior_ntbm = collect_graph_attention(
                    no_prior, graph_test_loader, device, tau
                )
                no_prior_C_j = []
                for s_ij, eidx, n_tbm in zip(no_prior_attn, no_prior_eidx, no_prior_ntbm):
                    if s_ij is not None:
                        C_j_ab, _ = aggregate_attention_to_surface(s_ij, eidx, n_tbm)
                        no_prior_C_j.append(C_j_ab)
                    else:
                        no_prior_C_j.append(None)

                no_prior_C_mean = np.array([
                    float(c.mean()) if c is not None and len(c) > 0 else 0.0
                    for c in no_prior_C_j
                ])
                no_prior_att_geo = compute_attention_geology_correlation(
                    no_prior_C_mean, test_chainages, geo_chainages, geo_values
                )

                moran_vals = []
                cv_vals = []
                for C_j_ab in no_prior_C_j:
                    if C_j_ab is not None and len(C_j_ab) == len(tbm_pos):
                        moran_vals.append(compute_morans_i(C_j_ab, tbm_pos))
                        cv_vals.append(compute_component_cv(C_j_ab, tbm_comp))

                ablation_spatial["No Geometric Prior"] = {
                    "attention_geology_r": no_prior_att_geo["pearson_r"],
                    "morans_i": float(np.mean(moran_vals)) if moran_vals else 0.0,
                    "component_cv": float(np.mean(cv_vals)) if cv_vals else 0.0,
                }
                logger.info(f"  No Geometric Prior: Moran's I={ablation_spatial['No Geometric Prior']['morans_i']:.3f}, "
                           f"CV={ablation_spatial['No Geometric Prior']['component_cv']:.3f}")
            except Exception as e:
                logger.warning(f"  Could not compute spatial consistency for No Geometry Prior: {e}")

        # Random Edges
        if "Random Edges" in ablation_results:
            try:
                rand_attn, rand_eidx, rand_ntbm = collect_graph_attention(
                    rand_model, graph_test_loader, device, tau
                )
                rand_C_j = []
                for s_ij, eidx, n_tbm in zip(rand_attn, rand_eidx, rand_ntbm):
                    if s_ij is not None:
                        C_j_ab, _ = aggregate_attention_to_surface(s_ij, eidx, n_tbm)
                        rand_C_j.append(C_j_ab)
                    else:
                        rand_C_j.append(None)

                rand_C_mean = np.array([
                    float(c.mean()) if c is not None and len(c) > 0 else 0.0
                    for c in rand_C_j
                ])
                rand_att_geo = compute_attention_geology_correlation(
                    rand_C_mean, test_chainages, geo_chainages, geo_values
                )

                moran_vals = []
                cv_vals = []
                for C_j_ab in rand_C_j:
                    if C_j_ab is not None and len(C_j_ab) == len(tbm_pos):
                        moran_vals.append(compute_morans_i(C_j_ab, tbm_pos))
                        cv_vals.append(compute_component_cv(C_j_ab, tbm_comp))

                ablation_spatial["Randomised Edges"] = {
                    "attention_geology_r": rand_att_geo["pearson_r"],
                    "morans_i": float(np.mean(moran_vals)) if moran_vals else 0.0,
                    "component_cv": float(np.mean(cv_vals)) if cv_vals else 0.0,
                }
                logger.info(f"  Randomised Edges: Moran's I={ablation_spatial['Randomised Edges']['morans_i']:.3f}, "
                           f"CV={ablation_spatial['Randomised Edges']['component_cv']:.3f}")
            except Exception as e:
                logger.warning(f"  Could not compute spatial consistency for Random Edges: {e}")

        spatial_consistency.update(ablation_spatial)

    # ── 分场景评估 ────────────────────────────────────────────────
    logger.info("Computing scenario-level metrics...")
    scenario_metrics_results = {}
    for mname, preds in all_preds.items():
        scenario_metrics_results[mname] = {}
        for sname, mask in scenario_masks.items():
            if mask.sum() >= 2:
                scenario_metrics_results[mname][sname] = compute_all_metrics(y_test[mask], preds[mask])
                scenario_metrics_results[mname][sname]["n_samples"] = int(mask.sum())

    # ── Bootstrap CI ──────────────────────────────────────────────
    logger.info("Computing bootstrap confidence intervals...")
    bootstrap_ci = {}
    for mname, preds in [("Full Model", full_preds), ("TSP-LSTM", tsp_lstm_preds),
                          ("LSTM", lstm_preds), ("SpatialAgg-LSTM", spa_preds)]:
        bootstrap_ci[mname] = bootstrap_metrics(y_test, preds, n_boot=runtime_cfg.get("bootstrap_samples", 300), seed=42)

    # ── 配对置换检验 (Full Model vs baselines) ────────────────────
    logger.info("Computing paired permutation tests...")
    perm_tests = {}
    for mname, preds in [("TSP-LSTM", tsp_lstm_preds), ("LSTM", lstm_preds),
                          ("SpatialAgg-LSTM", spa_preds)]:
        delta, p_val = paired_permutation_test(y_test, full_preds, preds,
                                                metric="mae", n_perm=runtime_cfg.get("permutation_samples", 2000), seed=42)
        perm_tests[f"Full vs {mname}"] = {"delta_mae": delta, "p_value": p_val}
        logger.info(f"  Full vs {mname}: ΔMAE={delta:.4f}, p={p_val:.4f}")

    # ── 逐变量指标 ────────────────────────────────────────────────
    per_var_metrics = {}
    for mname, preds in all_preds.items():
        per_var_metrics[mname] = compute_per_variable_metrics(y_test, preds)

    gen_fig1_framework(output_dir)
    gen_fig2_graph_construction(output_dir, snapshots, cfg, tbm_surface)
    gen_fig3_prediction(output_dir, test_chainages, y_test, all_preds)
    gen_fig4_ablation(output_dir, ablation_results, global_metrics)
    gen_fig5_hotspot(output_dir, test_chainages, test_graph_seqs,
                     attentions, edge_indices, n_tbm_list)
    gen_fig6_chainage_evolution(output_dir, test_chainages, C_mean_arr, y_test,
                                geo_chainages=geo_chainages, geo_values=geo_values)
    gen_fig7_decision_support(output_dir, test_chainages, y_test, full_preds,
                               C_mean_arr, geo_chainages, geo_values,
                               scenario_masks)
    gen_fig8_scenario_validation(output_dir, y_test, all_preds, scenario_masks)

    # ── Save metrics JSON ──────────────────────────────────────────
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_out = {k: {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else convert(v)
                   for k, v in global_metrics.items()}
    with open(output_dir / "metrics_global.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    ablation_out = {k: {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else convert(v)
                    for k, v in ablation_results.items()}
    with open(output_dir / "ablation_metrics.json", "w") as f:
        json.dump(ablation_out, f, indent=2)

    # 保存扩展分析结果
    extended_results = {
        "learned_beta": float(beta_val),
        "attention_geology_correlation": attn_geo_corr,
        "spatial_consistency": {
            mname: {kk: convert(vv) for kk, vv in v.items()}
            for mname, v in spatial_consistency.items()
        },
        "scenario_metrics": {
            mname: {sname: {kk: convert(vv) for kk, vv in v.items()}
                    for sname, v in scenarios.items()}
            for mname, scenarios in scenario_metrics_results.items()
        },
        "bootstrap_ci": {
            mname: {metric: {kk: convert(vv) for kk, vv in v.items()}
                    for metric, v in ci.items()}
            for mname, ci in bootstrap_ci.items()
        },
        "paired_permutation_tests": perm_tests,
        "per_variable_metrics": {
            mname: {vname: {kk: convert(vv) for kk, vv in v.items()}
                    for vname, v in var_metrics.items()}
            for mname, var_metrics in per_var_metrics.items()
        },
    }
    with open(output_dir / "extended_analysis.json", "w") as f:
        json.dump(extended_results, f, indent=2)

    logger.info(f"\nAll figures saved to {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
