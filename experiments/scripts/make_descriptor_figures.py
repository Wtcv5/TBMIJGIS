"""Make manuscript figures for explicit spatial interaction descriptors."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.style import (
    IJGIS_CMAPS,
    IJGIS_COLORS,
    add_panel_label,
    apply_ijgis_style,
    figure_size,
)


CASE_LABELS = {
    "bsll_dyk1017_205": "BSLL h=1",
    "bsll_dyk1017_205_h3": "BSLL h=3",
    "sjls_dyk1252_411": "SJLS h=3",
}

COMPONENT_ORDER = ["cutterhead", "front_shield", "middle_shield", "tail_shield"]
COMPONENT_LABELS = ["Cutterhead", "Front shield", "Middle shield", "Tail shield"]
COMPONENT_DISPLAY = dict(zip(COMPONENT_ORDER, COMPONENT_LABELS))
RESPONSE_ORDER = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
RESPONSE_LABELS = ["Advance\nrate", "Torque", "Thrust", "Penetr.", "Shield\npress."]
RESPONSE_DISPLAY = {
    "AdvanceRate": "advance rate",
    "Torque": "torque",
    "Thrust": "thrust",
    "Penetration": "penetration",
    "ShieldPressure": "shield pressure",
}

FIXED_RESPONSE_PAIRS = {
    "bsll_dyk1017_205": ("front_shield", "I_interaction_intensity", "AdvanceRate"),
    "bsll_dyk1017_205_h3": ("front_shield", "I_interaction_intensity", "ShieldPressure"),
    "sjls_dyk1252_411": ("cutterhead", "I_interaction_intensity", "ShieldPressure"),
}

VARIANT_LABELS = {
    "proposed": "Proposed",
    "chainage_only": "Chainage-only",
    "global_vp_anomaly": "Global Vp",
    "distance_only_exposure": "Distance-only",
    "uniform_edge_anomaly": "Uniform edge",
    "component_shuffle": "Component shuffle",
}

FIG_DPI = 600


def set_colorbar_style(cbar, label: str | None = None) -> None:
    """Apply the same typography and tick styling to all manuscript colorbars."""
    if label:
        cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)


def save_pdf_and_png(fig, pdf_path: Path) -> None:
    """Save editable vector PDF plus high-DPI PNG preview."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def box(ax, xy, text, width=0.34, height=0.28, fc="#F7F7F7", ec="#555555"):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=0.8, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=8)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->",
                                 mutation_scale=9, linewidth=0.8,
                                 color="#555555"))


def fixed_pair_rows(assoc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, (component, descriptor, response) in FIXED_RESPONSE_PAIRS.items():
        match = assoc[
            (assoc["case_id"] == case_id)
            & (assoc["component"] == component)
            & (assoc["descriptor"] == descriptor)
            & (assoc["response"] == response)
        ]
        if not match.empty:
            rows.append(match.iloc[0])
    return pd.DataFrame(rows)


def plot_method_framework(out_dir: Path) -> None:
    apply_ijgis_style()
    fig, axes = plt.subplots(1, 4, figsize=figure_size("double", aspect=0.27))
    panels = [
        ("Chainage-referenced entities",
         ["TSP rock voxels", "TBM surface components", "Monitoring responses"]),
        ("Geometry-constrained relations",
         ["Active zone", "Distance decay", "Normal compatibility"]),
        ("Component-chainage descriptors",
         [r"$A_c(t)$ supporting exposure", r"$I_c(t)$ anomaly intensity"]),
        ("Residual consistency",
         [r"$e_{t+h}=r_{t+h}-r_t$", "Descriptor-residual association"]),
    ]
    colors = ["#E8F1F6", "#F5EFE6", "#EEF3E6", "#F2ECF5"]
    for ax, (title, items), color, label in zip(axes, panels, colors, "abcd"):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8, pad=6)
        y_positions = np.linspace(0.68, 0.22, len(items))
        for y, item in zip(y_positions, items):
            box(ax, (0.12, y), item, width=0.76, height=0.16, fc=color)
        add_panel_label(ax, label, x=-0.02, y=1.02)
    for i in range(3):
        arrow(axes[i], (0.96, 0.5), (1.10, 0.5))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig1_method_framework.pdf")


def plot_spatial_entity_formalisation(out_dir: Path) -> None:
    apply_ijgis_style()
    fig, axes = plt.subplots(1, 3, figsize=figure_size("double", aspect=0.30))

    ax = axes[0]
    ax.set_title("Rock voxel field")
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.16, 0.84, size=(42, 2))
    vp = rng.uniform(0.15, 1.0, size=42)
    ax.scatter(xy[:, 0], xy[:, 1], c=vp, s=26, cmap=IJGIS_CMAPS["sequential"], edgecolor="white", linewidth=0.3)
    ax.arrow(0.12, 0.10, 0.70, 0.0, head_width=0.025, head_length=0.035, color="#555555", linewidth=0.8)
    ax.text(0.47, 0.04, "chainage", ha="center", va="center", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.set_title("TBM surface components")
    component_colors = ["#7A9E7E", "#D9A441", "#7C92B8", "#B96F6F"]
    xs = [0.16, 0.34, 0.52, 0.70]
    widths = [0.16, 0.18, 0.18, 0.18]
    for x0, w, color, label in zip(xs, widths, component_colors, COMPONENT_LABELS):
        ax.add_patch(Rectangle((x0, 0.42), w, 0.20, facecolor=color, edgecolor="#444444", linewidth=0.7))
        ax.text(x0 + w / 2, 0.32, label.replace(" ", "\n"), ha="center", va="top", fontsize=6.7)
    ax.add_patch(Circle((0.16, 0.52), 0.11, facecolor="#7A9E7E", edgecolor="#444444", linewidth=0.7))
    ax.arrow(0.12, 0.78, 0.70, 0.0, head_width=0.025, head_length=0.035, color="#555555", linewidth=0.8)
    ax.text(0.47, 0.85, "advance", ha="center", va="center", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    add_panel_label(ax, "b")

    ax = axes[2]
    ax.set_title("Component-chainage descriptors")
    chainage = np.arange(8)
    values = np.array([
        [0.20, 0.25, 0.40, 0.55, 0.78, 0.70, 0.48, 0.35],
        [0.14, 0.18, 0.28, 0.44, 0.62, 0.58, 0.40, 0.30],
        [0.10, 0.16, 0.20, 0.32, 0.46, 0.52, 0.38, 0.25],
        [0.06, 0.10, 0.15, 0.25, 0.36, 0.42, 0.31, 0.20],
    ])
    im = ax.imshow(values, aspect="auto", cmap=IJGIS_CMAPS["sequential"], vmin=0, vmax=0.8)
    ax.set_yticks(np.arange(4), COMPONENT_LABELS, fontsize=7)
    ax.set_xticks([0, 3, 7], [f"t+{i}" for i in [0, 3, 7]], fontsize=7)
    ax.set_xlabel("Target chainage step")
    ax.set_ylabel("Component")
    set_colorbar_style(fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02), "$I_c(t)$")
    add_panel_label(ax, "c")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig2_spatial_entity_formalisation.pdf")


def plot_geometry_constrained_edges(out_dir: Path) -> None:
    apply_ijgis_style()
    fig, axes = plt.subplots(1, 3, figsize=figure_size("double", aspect=0.30))

    ax = axes[0]
    ax.set_title(r"Active zone $\Omega_t$")
    ax.add_patch(Rectangle((0.10, 0.34), 0.72, 0.28, facecolor="#E8F1F6", edgecolor="#4A6D85", linewidth=0.8))
    ax.add_patch(Rectangle((0.43, 0.34), 0.25, 0.28, facecolor="#EEF3E6", edgecolor="#6F8F5D", linewidth=0.8))
    ax.add_patch(Circle((0.43, 0.48), 0.14, facecolor="#F5EFE6", edgecolor="#8B6F47", linewidth=0.8))
    ax.text(0.28, 0.70, r"$\Omega_t^{face}$", ha="center", fontsize=8)
    ax.text(0.57, 0.70, r"$\Omega_t^{shield}$", ha="center", fontsize=8)
    ax.arrow(0.12, 0.18, 0.68, 0.0, head_width=0.025, head_length=0.035, color="#555555", linewidth=0.8)
    ax.text(0.46, 0.11, "advance direction", ha="center", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.set_title("Distance and normal screening")
    ax.add_patch(Circle((0.50, 0.50), 0.18, facecolor="#F7F7F7", edgecolor="#555555", linewidth=0.8))
    ax.scatter([0.28, 0.40, 0.63, 0.75], [0.58, 0.30, 0.64, 0.42], s=24, color="#7A9E7E", edgecolor="white", linewidth=0.4)
    for x, y in [(0.40, 0.30), (0.63, 0.64)]:
        ax.plot([0.50, x], [0.50, y], color="#666666", linewidth=0.7)
    ax.arrow(0.50, 0.50, 0.18, 0.02, head_width=0.025, head_length=0.035, color="#B96F6F", linewidth=0.9)
    ax.text(0.62, 0.36, r"$d_{ij}\leq\tau$", fontsize=8)
    ax.text(0.56, 0.66, r"$\kappa_{ij}\geq\eta$", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    add_panel_label(ax, "b")

    ax = axes[2]
    ax.set_title("Weighted candidate relation")
    box(ax, (0.08, 0.58), "rock voxel\n$q_i$", width=0.25, height=0.18, fc="#EEF3E6")
    box(ax, (0.67, 0.58), "TBM node\ncomponent c", width=0.25, height=0.18, fc="#F5EFE6")
    arrow(ax, (0.34, 0.67), (0.66, 0.67))
    ax.text(0.50, 0.73, r"$w_{ij}^{rm}(t)$", ha="center", fontsize=8)
    box(ax, (0.24, 0.18), r"$A_c(t)=\sum w_{ij}$" + "\n" + r"$I_c(t)=\sum w_{ij}q_i/\sum w_{ij}$",
        width=0.52, height=0.22, fc="#F7F7F7")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    add_panel_label(ax, "c")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig3_geometry_constrained_edges.pdf")


def plot_case_context(out_dir: Path) -> None:
    apply_ijgis_style()
    rows = [
        {
            "case": "BSLL h=1",
            "tunnel": "BSLL DyK1017+205",
            "horizon": "h=1",
            "samples": "30 / 6 / 8",
            "interval": "41-48 m",
            "role": "compact diagnostic case",
        },
        {
            "case": "BSLL h=3",
            "tunnel": "BSLL DyK1017+205",
            "horizon": "h=3",
            "samples": "29 / 6 / 7",
            "interval": "42-48 m",
            "role": "compact multi-step case",
        },
        {
            "case": "SJLS h=3",
            "tunnel": "SJLS Dyk1252+411",
            "horizon": "h=3",
            "samples": "76 / 16 / 17",
            "interval": "99-115 m",
            "role": "external TSP contrast case",
        },
    ]
    fig, axes = plt.subplots(3, 1, figsize=figure_size("double", aspect=0.48), sharex=True)
    x_min, x_max = 0, 120
    colors = [IJGIS_COLORS["full_model"], IJGIS_COLORS["xgboost"], IJGIS_COLORS["lstm"]]
    for idx, (ax, row, color) in enumerate(zip(axes, rows, colors)):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.axhline(0.50, color="#D0D0D0", linewidth=1.0)
        start, end = [float(v) for v in row["interval"].replace(" m", "").split("-")]
        ax.add_patch(Rectangle((start, 0.36), end - start, 0.28, facecolor=color, alpha=0.28, edgecolor=color, linewidth=0.8))
        ax.text(1, 0.73, row["case"], ha="left", va="center", fontsize=8, fontweight="bold")
        ax.text(1, 0.48, row["tunnel"], ha="left", va="center", fontsize=7)
        ax.text(36, 0.73, f"{row['horizon']} | train/val/test {row['samples']}", ha="left", va="center", fontsize=7)
        ax.text(36, 0.48, f"test interval {row['interval']}", ha="left", va="center", fontsize=7)
        ax.text(76, 0.60, row["role"], ha="left", va="center", fontsize=7)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        add_panel_label(ax, "abc"[idx], x=-0.03, y=0.82)
    axes[-1].set_xlabel("Local target chainage (m)")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig5_case_context.pdf")


def plot_descriptor_evidence(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    fig = plt.figure(figsize=figure_size("double", aspect=0.58))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.85], hspace=0.22, wspace=0.18)
    heat_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    residual_axes = [fig.add_subplot(gs[1, i], sharex=heat_axes[i]) for i in range(3)]

    heatmaps = {}
    global_min = np.inf
    global_max = -np.inf
    for case_id in CASE_LABELS:
        case_df = read_csv(root / case_id / "component_spatial_descriptors.csv")
        heat = case_df.pivot(index="component", columns="chainage", values="I_interaction_intensity")
        heat = heat.loc[COMPONENT_ORDER]
        heatmaps[case_id] = heat
        global_min = min(global_min, float(np.nanmin(heat.values)))
        global_max = max(global_max, float(np.nanmax(heat.values)))

    im = None
    for idx, (ax_heat, ax_res, case_id) in enumerate(zip(heat_axes, residual_axes, CASE_LABELS)):
        heat = heatmaps[case_id]
        im = ax_heat.imshow(
            heat.values,
            aspect="auto",
            cmap=IJGIS_CMAPS["sequential"],
            vmin=global_min,
            vmax=global_max,
        )
        ax_heat.set_yticks(np.arange(len(COMPONENT_LABELS)), COMPONENT_LABELS if idx == 0 else [])
        cols = heat.columns.to_numpy(dtype=float)
        tick_idx = np.linspace(0, len(cols) - 1, min(4, len(cols)), dtype=int)
        ax_heat.set_xticks(tick_idx, [f"{cols[i]:.0f}" for i in tick_idx])
        ax_heat.tick_params(axis="x", labelbottom=False)
        ax_heat.set_title(CASE_LABELS.get(case_id, case_id))
        add_panel_label(ax_heat, "abc"[idx])

        component, _, response = FIXED_RESPONSE_PAIRS[case_id]
        case_df = read_csv(root / case_id / "component_spatial_descriptors.csv")
        comp_df = case_df[case_df["component"] == component].sort_values("chainage")
        res = comp_df[f"residual_{response}"].to_numpy(dtype=float)
        chainage = comp_df["chainage"].to_numpy(dtype=float)
        ax_res.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
        ax_res.plot(chainage, res, color=IJGIS_COLORS["truth"], marker="o", markersize=2.6, linewidth=1.0)
        ax_res.set_xlim(float(cols.min()), float(cols.max()))
        ax_res.set_xlabel("Chainage (m)")
        if idx == 0:
            ax_res.set_ylabel("Residual")
        ax_res.set_title(f"{component.replace('_', ' ')} vs {RESPONSE_DISPLAY[response]}", fontsize=7, pad=3)
    if im is not None:
        set_colorbar_style(
            fig.colorbar(im, ax=heat_axes, fraction=0.025, pad=0.02),
            "$I_c(t)$",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig6_descriptor_evidence.pdf")


def plot_sensitivity(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    paths = list(root.glob("*/sensitivity/descriptor_sensitivity_summary.csv"))
    if not paths:
        return
    fig, axes = plt.subplots(1, len(paths), figsize=figure_size("double", aspect=0.32), squeeze=False)
    for ax, path in zip(axes[0], paths):
        df = read_csv(path)
        case_id = path.parents[1].name
        pivot = df.pivot(index="eta_min", columns="tau_edge", values="fixed_spearman_r")
        im = ax.imshow(pivot.values, cmap=IJGIS_CMAPS["diverging"], vmin=-1, vmax=1, aspect="auto")
        component, _, response = FIXED_RESPONSE_PAIRS.get(case_id, ("cutterhead", "", "ShieldPressure"))
        ax.set_title(f"{CASE_LABELS.get(case_id, case_id)}\n{component.replace('_', ' ')} vs {RESPONSE_DISPLAY[response]}", fontsize=7)
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{i:g}" for i in pivot.index])
        ax.set_xlabel(r"$\tau_{edge}$")
        ax.set_ylabel(r"$\eta_{min}$")
    set_colorbar_style(
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02),
        "Fixed-pair $I_c$ Spearman rho",
    )
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig7_descriptor_sensitivity.pdf")


def plot_association_matrix_heatmap(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    assoc = read_csv(root / "descriptor_association_all.csv")
    assoc = assoc[assoc["descriptor"] == "I_interaction_intensity"].copy()
    fig, axes = plt.subplots(1, len(CASE_LABELS), figsize=figure_size("double", aspect=0.32), squeeze=False)
    im = None
    for ax, case_id in zip(axes[0], CASE_LABELS):
        case = assoc[assoc["case_id"] == case_id]
        matrix = (
            case.pivot(index="component", columns="response", values="spearman_r")
            .reindex(index=COMPONENT_ORDER, columns=RESPONSE_ORDER)
        )
        im = ax.imshow(matrix.values, cmap=IJGIS_CMAPS["diverging"], vmin=-1, vmax=1, aspect="auto")
        ax.set_title(CASE_LABELS.get(case_id, case_id))
        ax.set_xticks(np.arange(len(RESPONSE_LABELS)), RESPONSE_LABELS)
        ax.set_yticks(np.arange(len(COMPONENT_LABELS)), COMPONENT_LABELS)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="#222222")
    if im is not None:
        set_colorbar_style(
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02),
            "Spearman rho",
        )
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig8_descriptor_matrix_heatmap.pdf")


def plot_null_model_comparison(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    path = root / "descriptor_diagnostics" / "null_model_comparison.csv"
    if not path.exists():
        return
    df = read_csv(path)
    cases = list(CASE_LABELS)
    fig, axes = plt.subplots(1, len(cases), figsize=figure_size("double", aspect=0.34), squeeze=False)
    for ax, case_id in zip(axes[0], cases):
        case = df[df["case_id"] == case_id].copy()
        case["variant_label"] = case["variant"].map(VARIANT_LABELS)
        x = np.arange(len(case))
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.plot(x, case["spearman_r"], marker="o", color=IJGIS_COLORS["full_model"], label="Raw")
        ax.plot(x, case["detrended_spearman_r"], marker="s", color=IJGIS_COLORS["xgboost"], label="Detrended")
        ax.set_xticks(x, case["variant_label"], rotation=35, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(CASE_LABELS[case_id])
        if ax is axes[0][0]:
            ax.set_ylabel("Spearman rho")
        ax.grid(axis="y", alpha=0.25)
    axes[0][-1].legend(loc="lower right")
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig9_null_model_comparison.pdf")


def plot_traceability_example(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    path = root / "descriptor_diagnostics" / "top_contributing_edges_all.csv"
    if not path.exists():
        return
    df = read_csv(path)
    case = df[df["component"] == "cutterhead"].copy()
    if case.empty:
        case = df.copy()
    case = case.head(8)
    fig, axes = plt.subplots(1, 2, figsize=figure_size("double", aspect=0.36), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    norm = plt.Normalize(case["weighted_contribution"].min(), case["weighted_contribution"].max())
    cmap = plt.get_cmap(IJGIS_CMAPS["sequential"])
    for _, row in case.iterrows():
        color = cmap(norm(row["weighted_contribution"]))
        ax.plot([row["tbm_y"], row["rock_y"]], [row["tbm_z"], row["rock_z"]], color=color, linewidth=1.0, alpha=0.85)
    sc = ax.scatter(case["rock_y"], case["rock_z"], c=case["weighted_contribution"], cmap=IJGIS_CMAPS["sequential"],
                    s=34, edgecolor="white", linewidth=0.4, label="Rock voxel")
    ax.scatter(case["tbm_y"], case["tbm_z"], color=IJGIS_COLORS["tbm"], s=22, marker="s", label="TBM node")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local y (m)")
    ax.set_ylabel("Local z (m)")
    ax.set_title("Candidate edge trace")
    ax.legend(loc="upper right")
    add_panel_label(ax, "a")

    ax = axes[1]
    y = np.arange(len(case))[::-1]
    labels = [f"#{int(r)} rock {int(node)}" for r, node in zip(case["rank"], case["rock_node_id"])]
    ax.barh(y, case["weighted_contribution"], color=IJGIS_COLORS["full_model"], alpha=0.85)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"$w_{ij}q_i$")
    title_row = case.iloc[0]
    ax.set_title(f"{title_row['component'].replace('_', ' ')} at {title_row['chainage']:.0f} m")
    ax.grid(axis="x", alpha=0.25)
    add_panel_label(ax, "b")
    set_colorbar_style(fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02), r"$w_{ij}q_i$")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig10_traceability_example.pdf")


def main() -> None:
    exp_dir = Path(__file__).resolve().parent.parent
    root = exp_dir / "outputs" / "descriptors"
    out_dir = exp_dir / "outputs" / "figures"
    plot_method_framework(out_dir)
    plot_spatial_entity_formalisation(out_dir)
    plot_geometry_constrained_edges(out_dir)
    plot_case_context(out_dir)
    plot_descriptor_evidence(root, out_dir)
    plot_sensitivity(root, out_dir)
    plot_association_matrix_heatmap(root, out_dir)
    plot_null_model_comparison(exp_dir / "outputs", out_dir)
    plot_traceability_example(exp_dir / "outputs", out_dir)
    print(f"Saved descriptor figures to {out_dir}")


if __name__ == "__main__":
    main()
