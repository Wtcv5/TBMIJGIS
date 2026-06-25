"""Make manuscript figures for explicit spatial interaction descriptors."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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


def plot_method_framework(out_dir: Path) -> None:
    apply_ijgis_style()
    fig, axes = plt.subplots(1, 4, figsize=figure_size("double", aspect=0.27))
    panels = [
        ("Chainage-referenced entities",
         ["TSP rock voxels", "TBM surface components", "Monitoring responses"]),
        ("Geometry-constrained relations",
         ["Active zone", "Distance decay", "Normal compatibility"]),
        ("Component-chainage descriptors",
         [r"$A_c(t)$ geometric exposure", r"$I_c(t)$ anomaly intensity"]),
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


def plot_descriptor_evidence(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    fig = plt.figure(figsize=figure_size("double", aspect=0.72))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.28, hspace=0.34)
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_edges = fig.add_subplot(gs[1, 1])

    sjls = read_csv(root / "sjls_dyk1252_411" / "component_spatial_descriptors.csv")
    heat = sjls.pivot(index="component", columns="chainage", values="I_interaction_intensity")
    heat = heat.loc[COMPONENT_ORDER]
    im = ax_heat.imshow(heat.values, aspect="auto", cmap=IJGIS_CMAPS["sequential"])
    ax_heat.set_yticks(np.arange(len(COMPONENT_LABELS)), COMPONENT_LABELS)
    cols = heat.columns.to_numpy(dtype=float)
    tick_idx = np.linspace(0, len(cols) - 1, min(6, len(cols)), dtype=int)
    ax_heat.set_xticks(tick_idx, [f"{cols[i]:.0f}" for i in tick_idx])
    ax_heat.set_xlabel("Target chainage (m)")
    ax_heat.set_title("SJLS component-chainage interaction intensity $I_c(t)$")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.set_label("$I_c(t)$")
    add_panel_label(ax_heat, "a")

    assoc = read_csv(root / "descriptor_association_all.csv")
    assoc_i = assoc[assoc["descriptor"] == "I_interaction_intensity"].copy()
    best = []
    for case_id, group in assoc_i.groupby("case_id"):
        idx = group["spearman_r"].abs().idxmax()
        best.append(assoc_i.loc[idx])
    best_df = pd.DataFrame(best)
    y = np.arange(len(best_df))
    colors = [IJGIS_COLORS["accent"] if p < 0.05 else IJGIS_COLORS["persistence"] for p in best_df["spearman_p"]]
    ax_corr.axvline(0, color="#888888", linewidth=0.7)
    ax_corr.barh(y, best_df["spearman_r"], color=colors)
    labels = [
        f"{CASE_LABELS.get(r.case_id, r.case_id)}\n{r.component}, {r.response}"
        for r in best_df.itertuples()
    ]
    ax_corr.set_yticks(y, labels)
    ax_corr.set_xlabel("Spearman rho")
    ax_corr.set_title("Strongest $I_c(t)$--residual association")
    ax_corr.set_xlim(-1.0, 1.0)
    add_panel_label(ax_corr, "b")

    graph_rows = []
    for case_id in CASE_LABELS:
        p = root / case_id / "graph_construction_summary.csv"
        if not p.exists():
            continue
        g = read_csv(p)
        graph_rows.append({
            "case_id": case_id,
            "edges": g["total_candidate_edges"].mean(),
            "std": g["total_candidate_edges"].std(ddof=0),
        })
    gdf = pd.DataFrame(graph_rows)
    x = np.arange(len(gdf))
    ax_edges.bar(x, gdf["edges"], yerr=gdf["std"], color="#5B7C99", capsize=2)
    ax_edges.set_xticks(x, [CASE_LABELS.get(c, c) for c in gdf["case_id"]], rotation=20, ha="right")
    ax_edges.set_ylabel("Candidate edges per sample")
    ax_edges.set_title("Constructed rock-machine relation density")
    add_panel_label(ax_edges, "c")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, out_dir / "fig6_descriptor_evidence.pdf")
    # Save PNG preview too.
    plot_descriptor_evidence_png(root, out_dir)


def plot_descriptor_evidence_png(root: Path, out_dir: Path) -> None:
    # Re-run via PDF path helper would not create PNG sibling; keep explicit preview.
    apply_ijgis_style()
    fig = plt.figure(figsize=figure_size("double", aspect=0.72))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.28, hspace=0.34)
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_edges = fig.add_subplot(gs[1, 1])
    sjls = read_csv(root / "sjls_dyk1252_411" / "component_spatial_descriptors.csv")
    heat = sjls.pivot(index="component", columns="chainage", values="I_interaction_intensity").loc[COMPONENT_ORDER]
    im = ax_heat.imshow(heat.values, aspect="auto", cmap=IJGIS_CMAPS["sequential"])
    ax_heat.set_yticks(np.arange(len(COMPONENT_LABELS)), COMPONENT_LABELS)
    cols = heat.columns.to_numpy(dtype=float)
    tick_idx = np.linspace(0, len(cols) - 1, min(6, len(cols)), dtype=int)
    ax_heat.set_xticks(tick_idx, [f"{cols[i]:.0f}" for i in tick_idx])
    ax_heat.set_xlabel("Target chainage (m)")
    ax_heat.set_title("SJLS component-chainage interaction intensity $I_c(t)$")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02).set_label("$I_c(t)$")
    add_panel_label(ax_heat, "a")
    assoc = read_csv(root / "descriptor_association_all.csv")
    assoc_i = assoc[assoc["descriptor"] == "I_interaction_intensity"].copy()
    best = [group.loc[group["spearman_r"].abs().idxmax()] for _, group in assoc_i.groupby("case_id")]
    best_df = pd.DataFrame(best)
    y = np.arange(len(best_df))
    colors = [IJGIS_COLORS["accent"] if p < 0.05 else IJGIS_COLORS["persistence"] for p in best_df["spearman_p"]]
    ax_corr.axvline(0, color="#888888", linewidth=0.7)
    ax_corr.barh(y, best_df["spearman_r"], color=colors)
    ax_corr.set_yticks(y, [f"{CASE_LABELS.get(r.case_id, r.case_id)}\n{r.component}, {r.response}" for r in best_df.itertuples()])
    ax_corr.set_xlabel("Spearman rho")
    ax_corr.set_title("Strongest $I_c(t)$--residual association")
    ax_corr.set_xlim(-1.0, 1.0)
    add_panel_label(ax_corr, "b")
    graph_rows = []
    for case_id in CASE_LABELS:
        g = read_csv(root / case_id / "graph_construction_summary.csv")
        graph_rows.append({"case_id": case_id, "edges": g["total_candidate_edges"].mean(), "std": g["total_candidate_edges"].std(ddof=0)})
    gdf = pd.DataFrame(graph_rows)
    x = np.arange(len(gdf))
    ax_edges.bar(x, gdf["edges"], yerr=gdf["std"], color="#5B7C99", capsize=2)
    ax_edges.set_xticks(x, [CASE_LABELS.get(c, c) for c in gdf["case_id"]], rotation=20, ha="right")
    ax_edges.set_ylabel("Candidate edges per sample")
    ax_edges.set_title("Constructed rock-machine relation density")
    add_panel_label(ax_edges, "c")
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
        pivot = df.pivot(index="eta_min", columns="tau_edge", values="strongest_I_spearman_r")
        im = ax.imshow(pivot.values, cmap=IJGIS_CMAPS["diverging"], vmin=-1, vmax=1, aspect="auto")
        ax.set_title(CASE_LABELS.get(case_id, case_id))
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{i:g}" for i in pivot.index])
        ax.set_xlabel(r"$\tau_{edge}$")
        ax.set_ylabel(r"$\eta_{min}$")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Strongest $I_c$ Spearman rho")
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig7_descriptor_sensitivity.pdf", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig7_descriptor_sensitivity.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    exp_dir = Path(__file__).resolve().parent.parent
    root = exp_dir / "outputs" / "descriptors"
    out_dir = exp_dir / "outputs" / "figures"
    plot_method_framework(out_dir)
    plot_descriptor_evidence(root, out_dir)
    plot_sensitivity(root, out_dir)
    print(f"Saved descriptor figures to {out_dir}")


if __name__ == "__main__":
    main()
