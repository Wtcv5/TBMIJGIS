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
    CASE_PALETTE,
    COMPONENT_PALETTE,
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
    "component_reassignment": "Component reassign.",
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
    """Combined method figure: workflow pipeline (top) + spatial entity
    formalisation (bottom, three panels).
    """
    apply_ijgis_style()
    fig = plt.figure(figsize=figure_size("double", aspect=0.56))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.18)
    # --- Top: workflow pipeline ---
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(0, 12)
    ax_top.set_ylim(0, 4)
    ax_top.set_axis_off()

    stages = [
        {"xc": 1.5, "title": "1. Spatial entities",
         "lines": ["TSP rock voxels $D_{geo}$", "TBM surface $M_{TBM}$", "Chainage alignment"],
         "accent": IJGIS_COLORS["rock"]},
        {"xc": 4.5, "title": "2. Geometry-constrained graph",
         "lines": [r"Active zone $\Omega_t$", r"$d_{ij}\leq\tau_{edge}$", r"$\kappa_{ij}\geq\eta_{min}$"],
         "accent": IJGIS_COLORS["tbm"]},
        {"xc": 7.5, "title": "3. Component descriptors",
         "lines": [r"$A_c(t)=\sum w_{ij}$", r"$I_c(t)=\frac{\sum w_{ij}q_i}{\sum w_{ij}}$", "Per component $c$"],
         "accent": IJGIS_COLORS["accent"]},
        {"xc": 10.5, "title": "4. Residual diagnosis",
         "lines": [r"$e_{t+h}^{(k)}=r_{t+h}^{(k)}-r_t^{(k)}$", r"Spearman $\rho(I_c, e)$", "Null-model contrast"],
         "accent": IJGIS_COLORS["full_model"]},
    ]

    box_w, box_h = 2.6, 1.8
    y_box = 1.4
    for st in stages:
        ax_top.add_patch(FancyBboxPatch(
            (st["xc"] - box_w / 2, y_box), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=st["accent"], alpha=0.12, edgecolor=st["accent"], linewidth=1.2))
        ax_top.add_patch(Circle((st["xc"] - box_w / 2 + 0.30, y_box + box_h - 0.05), 0.16,
                            facecolor=st["accent"], edgecolor="white", linewidth=0.6, zorder=5))
        ax_top.text(st["xc"] - box_w / 2 + 0.30, y_box + box_h - 0.05,
                st["title"][0], ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", zorder=6)
        ax_top.text(st["xc"], y_box + box_h - 0.05, st["title"][3:],
                ha="center", va="center", fontsize=7.5, fontweight="bold", color=st["accent"])
        for j, line in enumerate(st["lines"]):
            ax_top.text(st["xc"], y_box + box_h - 0.55 - j * 0.35, line,
                    ha="center", va="center", fontsize=6.8)

    for i in range(3):
        x_start = stages[i]["xc"] + box_w / 2
        x_end = stages[i + 1]["xc"] - box_w / 2
        ax_top.annotate("", xy=(x_end, y_box + box_h / 2), xytext=(x_start, y_box + box_h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=IJGIS_COLORS["gray_dark"],
                                    lw=1.1, mutation_scale=12))

    ax_top.text(stages[0]["xc"], 0.7, "Input\nGeological + monitoring",
            ha="center", va="center", fontsize=6.5, style="italic",
            color=IJGIS_COLORS["gray_dark"])
    ax_top.text(stages[-1]["xc"], 0.7, "Output\nDescriptor evidence",
            ha="center", va="center", fontsize=6.5, style="italic",
            color=IJGIS_COLORS["gray_dark"])

    ax_top.text(6.0, 3.6, "Chainage-referenced rock–TBM spatial interaction graph: workflow",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=IJGIS_COLORS["truth"])
    ax_top.plot([0.5, 11.5], [3.25, 3.25], color=IJGIS_COLORS["gray_light"], linewidth=0.5)
    add_panel_label(ax_top, "a", x=-0.02, y=1.05)

    # --- Bottom: spatial entity formalisation (three panels) ---
    gs_bottom = gs[1].subgridspec(1, 3, wspace=0.28)
    ax_b0 = fig.add_subplot(gs_bottom[0, 0])
    ax_b0.set_title("Rock voxel field", fontsize=8)
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.16, 0.84, size=(42, 2))
    vp = rng.uniform(0.15, 1.0, size=42)
    ax_b0.scatter(xy[:, 0], xy[:, 1], c=vp, s=26, cmap=IJGIS_CMAPS["sequential"], edgecolor="white", linewidth=0.3)
    ax_b0.arrow(0.12, 0.10, 0.70, 0.0, head_width=0.025, head_length=0.035, color="#555555", linewidth=0.8)
    ax_b0.text(0.47, 0.04, "chainage", ha="center", va="center", fontsize=7)
    ax_b0.set_xlim(0, 1)
    ax_b0.set_ylim(0, 1)
    ax_b0.set_axis_off()
    add_panel_label(ax_b0, "b", x=-0.04, y=1.02)

    ax_b1 = fig.add_subplot(gs_bottom[0, 1])
    ax_b1.set_title("TBM surface components", fontsize=8)
    component_colors = [COMPONENT_PALETTE[c] for c in COMPONENT_ORDER]
    xs = [0.16, 0.34, 0.52, 0.70]
    widths = [0.16, 0.18, 0.18, 0.18]
    for x0, w, color, label in zip(xs, widths, component_colors, COMPONENT_LABELS):
        ax_b1.add_patch(Rectangle((x0, 0.42), w, 0.20, facecolor=color, edgecolor="#444444", linewidth=0.7))
        ax_b1.text(x0 + w / 2, 0.32, label.replace(" ", "\n"), ha="center", va="top", fontsize=6.7)
    ax_b1.add_patch(Circle((0.16, 0.52), 0.11, facecolor="#7A9E7E", edgecolor="#444444", linewidth=0.7))
    ax_b1.arrow(0.12, 0.78, 0.70, 0.0, head_width=0.025, head_length=0.035, color="#555555", linewidth=0.8)
    ax_b1.text(0.47, 0.85, "advance", ha="center", va="center", fontsize=7)
    ax_b1.set_xlim(0, 1)
    ax_b1.set_ylim(0, 1)
    ax_b1.set_axis_off()
    add_panel_label(ax_b1, "c", x=-0.04, y=1.02)

    ax_b2 = fig.add_subplot(gs_bottom[0, 2])
    ax_b2.set_title("Component-chainage descriptors", fontsize=8)
    values = np.array([
        [0.20, 0.25, 0.40, 0.55, 0.78, 0.70, 0.48, 0.35],
        [0.14, 0.18, 0.28, 0.44, 0.62, 0.58, 0.40, 0.30],
        [0.10, 0.16, 0.20, 0.32, 0.46, 0.52, 0.38, 0.25],
        [0.06, 0.10, 0.15, 0.25, 0.36, 0.42, 0.31, 0.20],
    ])
    im = ax_b2.imshow(values, aspect="auto", cmap=IJGIS_CMAPS["sequential"], vmin=0, vmax=0.8)
    ax_b2.set_yticks(np.arange(4), COMPONENT_LABELS, fontsize=7)
    ax_b2.set_xticks([0, 3, 7], [f"t+{i}" for i in [0, 3, 7]], fontsize=7)
    ax_b2.set_xlabel("Target chainage step", fontsize=7)
    ax_b2.set_ylabel("Component", fontsize=7)
    set_colorbar_style(fig.colorbar(im, ax=ax_b2, fraction=0.046, pad=0.02), "$I_c(t)$")
    add_panel_label(ax_b2, "d", x=-0.12, y=1.02)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig1_method_framework.pdf")


def plot_spatial_entity_formalisation(out_dir: Path) -> None:
    """Deprecated: merged into plot_method_framework. Kept as no-op to avoid
    breaking call sites; the combined figure is written by plot_method_framework.
    """
    return


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
        ax.plot([0.50, x], [0.50, y], color=IJGIS_COLORS["gray_medium"], linewidth=0.7)
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


def plot_case_context(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    case_meta = {
        "bsll_dyk1017_205": {"label": "BSLL h=1", "tunnel": "DyK1017+205",
                              "interval": (41.0, 48.0), "samples": "30 / 6 / 8"},
        "bsll_dyk1017_205_h3": {"label": "BSLL h=3", "tunnel": "DyK1017+205",
                                 "interval": (42.0, 48.0), "samples": "29 / 6 / 7"},
        "sjls_dyk1252_411": {"label": "SJLS h=3", "tunnel": "Dyk1252+411",
                              "interval": (99.0, 115.0), "samples": "76 / 16 / 17"},
    }
    fig, axes = plt.subplots(3, 1, figsize=figure_size("double", aspect=0.52), sharex=False)
    comp_colors = [COMPONENT_PALETTE[c] for c in COMPONENT_ORDER]
    for idx, (ax, (case_id, meta)) in enumerate(zip(axes, case_meta.items())):
        csv_path = root / case_id / "component_spatial_descriptors.csv"
        if not csv_path.exists():
            ax.set_axis_off()
            continue
        df = read_csv(csv_path)
        # Test interval shading
        t_start, t_end = meta["interval"]
        ax.axvspan(t_start, t_end, color=CASE_PALETTE[case_id], alpha=0.10, zorder=0)
        ax.axvline(t_start, color=CASE_PALETTE[case_id], linewidth=0.6, linestyle=":", zorder=1)
        ax.axvline(t_end, color=CASE_PALETTE[case_id], linewidth=0.6, linestyle=":", zorder=1)
        # Plot I_c(t) per component vs chainage
        for comp, color, comp_label in zip(COMPONENT_ORDER, comp_colors, COMPONENT_LABELS):
            cdf = df[df["component"] == comp].sort_values("chainage")
            ax.plot(cdf["chainage"], cdf["I_interaction_intensity"],
                    color=color, linewidth=1.0, marker="o", markersize=2.2,
                    label=comp_label, zorder=3)
        # Case label (annotated on plot)
        ax.text(0.02, 0.95, f"{meta['label']}  ({meta['tunnel']})",
                transform=ax.transAxes, fontsize=7.5, fontweight="bold",
                color=CASE_PALETTE[case_id], va="top")
        ax.text(0.98, 0.95, f"train/val/test: {meta['samples']}",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
                color=IJGIS_COLORS["gray_dark"])
        ax.set_ylabel(r"$I_c(t)$", fontsize=7)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=6, ncol=2, frameon=False)
        if idx == len(case_meta) - 1:
            ax.set_xlabel("Chainage (m)")
        add_panel_label(ax, "abc"[idx], x=-0.04, y=0.98)
    fig.subplots_adjust(hspace=0.38)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig4_case_context.pdf")


def plot_descriptor_evidence(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    fig = plt.figure(figsize=figure_size("double", aspect=0.54))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.30)
    desc_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    residual_axes = [fig.add_subplot(gs[1, i], sharex=desc_axes[i]) for i in range(3)]

    for idx, (ax_desc, ax_res, case_id) in enumerate(zip(desc_axes, residual_axes, CASE_LABELS)):
        component, _, response = FIXED_RESPONSE_PAIRS[case_id]
        case_df = read_csv(root / case_id / "component_spatial_descriptors.csv")
        comp_df = case_df[case_df["component"] == component].sort_values("chainage")
        chainage = comp_df["chainage"].to_numpy(dtype=float)
        descriptor = comp_df["I_interaction_intensity"].to_numpy(dtype=float)
        res = comp_df[f"residual_{response}"].to_numpy(dtype=float)

        comp_color = COMPONENT_PALETTE.get(component, IJGIS_COLORS["full_model"])
        ax_desc.plot(chainage, descriptor, color=comp_color, marker="o", markersize=2.8, linewidth=1.1)
        ax_desc.set_title(f"{CASE_LABELS.get(case_id, case_id)}\n{component.replace('_', ' ')} / {RESPONSE_DISPLAY[response].replace(chr(10), ' ')}", fontsize=7, pad=3)
        if idx == 0:
            ax_desc.set_ylabel(r"$I_c(t)$")
        ax_desc.grid(axis="y", alpha=0.25)
        ax_desc.tick_params(axis="x", labelbottom=False)
        ax_desc.spines["top"].set_visible(False)
        ax_desc.spines["right"].set_visible(False)
        add_panel_label(ax_desc, "abc"[idx])

        ax_res.axhline(0, color=IJGIS_COLORS["gray_medium"], linewidth=0.6, linestyle=":")
        ax_res.plot(chainage, res, color=IJGIS_COLORS["truth"], marker="o", markersize=2.6, linewidth=1.0)
        ax_res.set_xlim(float(chainage.min()), float(chainage.max()))
        ax_res.set_xlabel("Chainage (m)")
        if idx == 0:
            ax_res.set_ylabel("Residual")
        ax_res.grid(axis="y", alpha=0.25)
        ax_res.spines["top"].set_visible(False)
        ax_res.spines["right"].set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig5_descriptor_evidence.pdf")


def plot_sensitivity(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    paths = sorted(
        root.glob("*/sensitivity/descriptor_sensitivity_summary.csv"),
        key=lambda p: list(CASE_LABELS).index(p.parents[1].name) if p.parents[1].name in CASE_LABELS else 99,
    )
    if not paths:
        return
    fig, axes = plt.subplots(
        1,
        len(paths),
        figsize=figure_size("double", aspect=0.34),
        squeeze=False,
        constrained_layout=True,
    )
    im = None
    for idx, (ax, path) in enumerate(zip(axes[0], paths)):
        df = read_csv(path)
        case_id = path.parents[1].name
        pivot = (
            df.pivot(index="eta_min", columns="tau_edge", values="fixed_spearman_r")
            .sort_index()
            .sort_index(axis=1)
        )
        im = ax.imshow(pivot.values, cmap=IJGIS_CMAPS["diverging"], vmin=-1, vmax=1, aspect="auto")
        component, _, response = FIXED_RESPONSE_PAIRS.get(case_id, ("cutterhead", "", "ShieldPressure"))
        pair_label = f"{component.replace('_', ' ')} / {RESPONSE_DISPLAY[response].replace(chr(10), ' ')}"
        ax.set_title(f"{CASE_LABELS.get(case_id, case_id)}\n{pair_label}", fontsize=7, pad=3)
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        if idx == 0:
            ax.set_yticks(np.arange(len(pivot.index)), [f"{i:g}" for i in pivot.index])
            ax.set_ylabel(r"$\eta_{\min}$")
        else:
            ax.set_yticks(np.arange(len(pivot.index)), [])
        ax.set_xlabel(r"$\tau_{edge}$")
        ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                color = "white" if abs(value) > 0.55 else "#222222"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.2, color=color)
        if 2.0 in pivot.columns and 0.3 in pivot.index:
            j = list(pivot.columns).index(2.0)
            i = list(pivot.index).index(0.3)
            rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#111111", linewidth=1.0)
            ax.add_patch(rect)
    set_colorbar_style(
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02),
        r"Fixed-pair Spearman $\rho$",
    )
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig7_descriptor_sensitivity.pdf")


def plot_null_model_comparison(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    path = root / "descriptor_diagnostics" / "null_model_comparison.csv"
    if not path.exists():
        return
    df = read_csv(path)
    cases = list(CASE_LABELS)
    fig, axes = plt.subplots(1, len(cases), figsize=figure_size("double", aspect=0.42), squeeze=False)
    bars_handles = []
    for ax, case_id in zip(axes[0], cases):
        case = df[df["case_id"] == case_id].copy()
        case["variant_label"] = case["variant"].map(VARIANT_LABELS)
        x = np.arange(len(case))
        ax.axhline(0, color=IJGIS_COLORS["gray_medium"], linewidth=0.6)
        width = 0.38
        b1 = ax.bar(x - width / 2, case["spearman_r"], width=width, color=IJGIS_COLORS["diagnostic_raw"], alpha=0.85, label="Raw")
        b2 = ax.bar(x + width / 2, case["detrended_spearman_r"], width=width, color=IJGIS_COLORS["diagnostic_detrended"], alpha=0.85, label="Detrended")
        if not bars_handles:
            bars_handles = [b1, b2]
        ax.set_xticks(x, case["variant_label"], rotation=40, ha="right", fontsize=6.2)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(CASE_LABELS[case_id])
        ax.set_ylabel("Spearman rho")
        ax.grid(axis="y", alpha=0.25)
    # Legend outside data area (above subplots, centred)
    fig.legend(bars_handles, ["Raw", "Detrended"], loc="upper center",
               ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    for label, ax in zip("abc", axes[0]):
        add_panel_label(ax, label, x=-0.16, y=1.08)
    fig.subplots_adjust(top=0.88, bottom=0.22, wspace=0.32)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig6_null_model_comparison.pdf")


def plot_traceability_example(root: Path, out_dir: Path) -> None:
    apply_ijgis_style()
    path = root / "descriptor_diagnostics" / "top_contributing_edges_all.csv"
    if not path.exists():
        return
    df = read_csv(path)
    case = df[df["component"] == "cutterhead"].copy()
    if case.empty:
        case = df.copy()
    case = case.head(5).copy()
    top3 = case.head(3).copy()
    fig = plt.figure(figsize=figure_size("double", aspect=0.42))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    norm = plt.Normalize(case["weighted_contribution"].min(), case["weighted_contribution"].max())
    cmap = plt.get_cmap(IJGIS_CMAPS["sequential"])
    for _, row in top3.iterrows():
        color = cmap(norm(row["weighted_contribution"]))
        ax.plot([row["tbm_y"], row["rock_y"]], [row["tbm_z"], row["rock_z"]],
                color=color, linewidth=1.5, alpha=0.90)
        mid_y = 0.52 * row["rock_y"] + 0.48 * row["tbm_y"]
        mid_z = 0.52 * row["rock_z"] + 0.48 * row["tbm_z"]
        ax.text(mid_y, mid_z, f"#{int(row['rank'])}", fontsize=6.5, ha="center",
                va="center", color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="none", alpha=0.95))
    sc = ax.scatter(top3["rock_y"], top3["rock_z"], c=top3["weighted_contribution"],
                    cmap=IJGIS_CMAPS["sequential"], s=42, edgecolor="white",
                    linewidth=0.4, label="Rock voxel")
    ax.scatter(top3["tbm_y"], top3["tbm_z"], color=IJGIS_COLORS["tbm"], s=28,
               marker="s", label="TBM node")
    y_min = min(top3["rock_z"].min(), top3["tbm_z"].min())
    y_max = max(top3["rock_z"].max(), top3["tbm_z"].max())
    x_min = min(top3["rock_y"].min(), top3["tbm_y"].min())
    x_max = max(top3["rock_y"].max(), top3["tbm_y"].max())
    ax.set_xlim(x_min - 0.55, x_max + 0.55)
    ax.set_ylim(y_min - 0.55, y_max + 0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Cross-section $y$ (m)", fontsize=7)
    ax.set_ylabel("Cross-section $z$ (m)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.2)
    ax.set_title("Top-3 spatial edge trace", fontsize=8, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=6, frameon=True, framealpha=0.9, edgecolor="none")
    add_panel_label(ax, "a", x=-0.08, y=1.08)

    ax = fig.add_subplot(gs[0, 1])
    title_row = case.iloc[0]
    y = np.arange(len(case))[::-1]
    labels = [f"#{int(r)}  v{int(v)} -> n{int(n)}" for r, v, n in zip(case["rank"], case["rock_node_id"], case["tbm_node_id"])]
    ax.barh(y, case["weighted_contribution"], color=COMPONENT_PALETTE.get(title_row["component"], IJGIS_COLORS["gray_medium"]), alpha=0.85)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"$w_{ij}q_i$")
    ax.set_title(f"Top-5 edge contributions: {title_row['component'].replace('_', ' ')} at {title_row['chainage']:.0f} m", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, "b", x=-0.08, y=1.08)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, out_dir / "fig8_traceability_example.pdf")


def main() -> None:
    exp_dir = Path(__file__).resolve().parent.parent
    root = exp_dir / "outputs" / "descriptors"
    out_dir = exp_dir / "outputs" / "figures"
    plot_method_framework(out_dir)
    plot_spatial_entity_formalisation(out_dir)
    plot_geometry_constrained_edges(out_dir)
    plot_case_context(root, out_dir)
    plot_descriptor_evidence(root, out_dir)
    plot_sensitivity(root, out_dir)
    plot_null_model_comparison(exp_dir / "outputs", out_dir)
    plot_traceability_example(exp_dir / "outputs", out_dir)
    print(f"Saved descriptor figures to {out_dir}")


if __name__ == "__main__":
    main()
