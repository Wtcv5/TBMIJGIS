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
    save_publication_figure,
)


CASE_LABELS = {
    "bsll_dyk1017_205": "BSLL h=1",
    "bsll_dyk1017_205_h3": "BSLL h=3",
    "sjls_dyk1252_411": "SJLS h=1",
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
    fig.savefig(out_dir / "fig1_method_framework.pdf", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_method_framework.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("$I_c(t)$")
    add_panel_label(ax, "c")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_spatial_entity_formalisation.pdf", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig2_spatial_entity_formalisation.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    fig.savefig(out_dir / "fig3_geometry_constrained_edges.pdf", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig3_geometry_constrained_edges.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_descriptor_evidence(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    fig, heat_axes = plt.subplots(1, 3, figsize=figure_size("double", aspect=0.34), squeeze=False)
    heat_axes = heat_axes[0]

    for idx, (ax_heat, case_id) in enumerate(zip(heat_axes, CASE_LABELS)):
        case_df = read_csv(root / case_id / "component_spatial_descriptors.csv")
        heat = case_df.pivot(index="component", columns="chainage", values="I_interaction_intensity")
        heat = heat.loc[COMPONENT_ORDER]
        im = ax_heat.imshow(heat.values, aspect="auto", cmap=IJGIS_CMAPS["sequential"])
        ax_heat.set_yticks(np.arange(len(COMPONENT_LABELS)), COMPONENT_LABELS if idx == 0 else [])
        cols = heat.columns.to_numpy(dtype=float)
        tick_idx = np.linspace(0, len(cols) - 1, min(4, len(cols)), dtype=int)
        ax_heat.set_xticks(tick_idx, [f"{cols[i]:.0f}" for i in tick_idx])
        ax_heat.set_xlabel("Chainage (m)")
        ax_heat.set_title(CASE_LABELS.get(case_id, case_id))
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
        if idx == len(heat_axes) - 1:
            cbar.set_label("$I_c(t)$")
        add_panel_label(ax_heat, "abc"[idx])

    out_dir.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, out_dir / "fig6_descriptor_evidence.pdf")
    # Save PNG preview too.
    plot_descriptor_evidence_png(root, out_dir)


def plot_descriptor_evidence_png(root: Path, out_dir: Path) -> None:
    # Re-run via PDF path helper would not create PNG sibling; keep explicit preview.
    apply_ijgis_style()
    fig, heat_axes = plt.subplots(1, 3, figsize=figure_size("double", aspect=0.34), squeeze=False)
    heat_axes = heat_axes[0]
    for idx, (ax_heat, case_id) in enumerate(zip(heat_axes, CASE_LABELS)):
        case_df = read_csv(root / case_id / "component_spatial_descriptors.csv")
        heat = case_df.pivot(index="component", columns="chainage", values="I_interaction_intensity").loc[COMPONENT_ORDER]
        im = ax_heat.imshow(heat.values, aspect="auto", cmap=IJGIS_CMAPS["sequential"])
        ax_heat.set_yticks(np.arange(len(COMPONENT_LABELS)), COMPONENT_LABELS if idx == 0 else [])
        cols = heat.columns.to_numpy(dtype=float)
        tick_idx = np.linspace(0, len(cols) - 1, min(4, len(cols)), dtype=int)
        ax_heat.set_xticks(tick_idx, [f"{cols[i]:.0f}" for i in tick_idx])
        ax_heat.set_xlabel("Chainage (m)")
        ax_heat.set_title(CASE_LABELS.get(case_id, case_id))
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
        if idx == len(heat_axes) - 1:
            cbar.set_label("$I_c(t)$")
        add_panel_label(ax_heat, "abc"[idx])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig6_descriptor_evidence.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
        ax.set_title(CASE_LABELS.get(case_id, case_id))
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{i:g}" for i in pivot.index])
        ax.set_xlabel(r"$\tau_{edge}$")
        ax.set_ylabel(r"$\eta_{min}$")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Fixed-pair $I_c$ Spearman rho")
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig7_descriptor_sensitivity.pdf", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig7_descriptor_sensitivity.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Spearman rho")
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig8_descriptor_matrix_heatmap.pdf", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig8_descriptor_matrix_heatmap.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    exp_dir = Path(__file__).resolve().parent.parent
    root = exp_dir / "outputs" / "descriptors"
    out_dir = exp_dir / "outputs" / "figures"
    plot_method_framework(out_dir)
    plot_spatial_entity_formalisation(out_dir)
    plot_geometry_constrained_edges(out_dir)
    plot_descriptor_evidence(root, out_dir)
    plot_sensitivity(root, out_dir)
    plot_association_matrix_heatmap(root, out_dir)
    print(f"Saved descriptor figures to {out_dir}")


if __name__ == "__main__":
    main()
