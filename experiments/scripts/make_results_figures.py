"""Generate case and result figures for the manuscript.

This script reads saved experiment summaries and post-hoc evidence only. It
does not retrain models or invent predictions that are not available.

Usage:
    python scripts/make_results_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.style import IJGIS_COLORS, add_panel_label, apply_ijgis_style


EXP_DIR = Path(__file__).resolve().parents[1]
OUT_DEFAULT = Path("outputs/figures")

COL = {
    "ink": "#263238",
    "muted": "#607D8B",
    "grid": "#D7DEE2",
    "blue": IJGIS_COLORS["full_model"],
    "teal": IJGIS_COLORS["lstm"],
    "orange": IJGIS_COLORS["xgboost"],
    "purple": IJGIS_COLORS["accent"],
    "gray": IJGIS_COLORS["persistence"],
    "rock": IJGIS_COLORS["rock"],
    "tbm": IJGIS_COLORS["tbm"],
    "light_blue": "#E8F2F8",
    "light_teal": "#E6F3F1",
    "light_orange": "#FFF2E5",
    "light_purple": "#F1EAF6",
}

COMPONENTS = ["cutterhead", "front_shield", "middle_shield", "tail_shield"]
COMP_LABELS = ["Cutterhead", "Front shield", "Middle shield", "Tail shield"]
COMP_COLORS = {
    "cutterhead": "#8DA0CB",
    "front_shield": "#66C2A5",
    "middle_shield": "#FC8D62",
    "tail_shield": "#A6D854",
}


def save_pdf_png(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def round_box(ax, xy, w, h, text, face, edge, fs=7.0, weight="normal"):
    box = patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=0.8,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=COL["ink"], fontweight=weight, linespacing=1.15)


def arrow(ax, start, end, color=COL["muted"], lw=0.8, rad=0.0):
    ax.add_patch(patches.FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2,
    ))


def load_case_summary() -> pd.DataFrame:
    return pd.read_csv(EXP_DIR / "outputs" / "case_summary.csv")


def load_interp_summary() -> pd.DataFrame:
    return pd.read_csv(EXP_DIR / "outputs" / "evidence" / "interpretation_summary.csv")


def load_component(case_id: str) -> pd.DataFrame:
    return pd.read_csv(EXP_DIR / "outputs" / "evidence" / case_id / "component_relevance.csv")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def case_label(case_id: str) -> str:
    return {
        "bsll_dyk1017_205": "BSLL h=1",
        "bsll_dyk1017_205_h3": "BSLL h=3",
        "sjls_dyk1252_411": "SJLS h=1",
    }.get(case_id, case_id)


def fig5_case_context(out_dir: Path, cases: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10.8, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[0.88, 1.0], wspace=0.18, hspace=0.28)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        clean_axis(ax)

    fig.suptitle("Two real tunnel scenarios instantiate the same transferable method",
                 fontsize=10, fontweight="bold", y=0.98)

    # a: BSLL case role
    ax = axes[0]
    ax.set_title("BSLL DyK1017+205", loc="left", fontweight="bold")
    round_box(ax, (0.05, 0.63), 0.32, 0.18, "compact\ncase interval", COL["light_blue"], COL["blue"], weight="bold")
    round_box(ax, (0.43, 0.63), 0.22, 0.18, "h=1\nn=30/6/8", "#FFFFFF", COL["gray"], weight="bold")
    round_box(ax, (0.71, 0.63), 0.22, 0.18, "h=3\nn=29/6/7", "#FFFFFF", COL["purple"], weight="bold")
    ax.plot([0.10, 0.90], [0.35, 0.35], color=COL["ink"], linewidth=0.9)
    ax.add_patch(patches.Rectangle((0.18, 0.42), 0.55, 0.10, facecolor=COL["light_blue"], edgecolor=COL["rock"], linewidth=0.8))
    ax.text(0.455, 0.47, "project TSP voxel field", ha="center", va="center", fontsize=6.8, color=COL["rock"])
    ax.add_patch(patches.Rectangle((0.62, 0.25), 0.22, 0.20, facecolor="#FFFFFF", edgecolor=COL["purple"], linewidth=0.9))
    ax.text(0.73, 0.21, "test: 41/42-48 m", ha="center", fontsize=6.6)
    ax.text(0.50, 0.08, "DyK1017+205", ha="center", fontsize=7.0, color=COL["muted"])
    add_panel_label(ax, "a")

    # b: SJLS case role
    ax = axes[1]
    ax.set_title("SJLS Dyk1252+411", loc="left", fontweight="bold")
    round_box(ax, (0.06, 0.63), 0.34, 0.18, "external\nTSP case", COL["light_teal"], COL["teal"], weight="bold")
    round_box(ax, (0.49, 0.63), 0.22, 0.18, "h=1\nn=76/16/17", "#FFFFFF", COL["gray"], weight="bold")
    ax.plot([0.10, 0.90], [0.35, 0.35], color=COL["ink"], linewidth=0.9)
    ax.add_patch(patches.Rectangle((0.16, 0.42), 0.62, 0.10, facecolor=COL["light_teal"], edgecolor=COL["teal"], linewidth=0.8))
    ax.text(0.47, 0.47, "dense P/S-wave velocity profiles", ha="center", va="center", fontsize=6.8, color=COL["teal"])
    ax.add_patch(patches.Rectangle((0.57, 0.25), 0.28, 0.20, facecolor="#FFFFFF", edgecolor=COL["blue"], linewidth=0.9))
    ax.text(0.71, 0.21, "test: 99-115 m", ha="center", fontsize=6.6)
    ax.text(0.50, 0.08, "Dyk1252+411", ha="center", fontsize=7.0, color=COL["muted"])
    add_panel_label(ax, "b")

    # c: sample and split summary
    ax = axes[2]
    ax.set_title("Mileage-ordered evaluation", loc="left", fontweight="bold")
    labels = [case_label(x) for x in cases["case_id"]]
    y = np.arange(len(labels))
    ax.set_axis_on()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Effective samples")
    val_samples = []
    for case_id in cases["case_id"]:
        audit = load_json(EXP_DIR / "outputs" / case_id / "preprocessing_audit.json")
        val_samples.append(len(audit["val_sample_indices"]))
    val_samples = np.asarray(val_samples)
    train_samples = cases["train_samples"].to_numpy()
    test_samples = cases["test_samples"].to_numpy()
    ax.barh(y, train_samples, color=COL["blue"], alpha=0.8, label="train")
    ax.barh(y, val_samples, left=train_samples, color=COL["teal"], alpha=0.78, label="validation")
    ax.barh(y, test_samples, left=train_samples + val_samples, color=COL["orange"], alpha=0.85, label="test")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    add_panel_label(ax, "c")

    # d: plain case overview
    ax = axes[3]
    ax.set_title("Case overview", loc="left", fontweight="bold")
    rows = [
        ("TSP source", "project voxel field", "external velocity field"),
        ("Formal horizons", "h=1 and h=3", "h=1"),
        ("Test interval", "8 / 7 samples", "17 samples"),
        ("Experimental use", "multi-step diagnostics", "geometry ablation check"),
    ]
    ax.set_axis_on()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(rows))
    ax.set_xticks([1.35, 2.35], ["BSLL", "SJLS"])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for i, (row_name, bsll, sjls) in enumerate(rows):
        y0 = len(rows) - 1 - i
        ax.text(0.40, y0 + 0.5, row_name, ha="center", va="center",
                fontsize=6.7, color=COL["ink"])
        for j, txt in enumerate([bsll, sjls]):
            x0 = 0.85 + j
            ax.add_patch(patches.Rectangle((x0, y0), 1.0, 1.0,
                                           facecolor="#F7F9FA",
                                           edgecolor="white", linewidth=1))
            ax.text(x0 + 0.5, y0 + 0.5, txt, ha="center", va="center",
                    fontsize=6.7, color=COL["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "d")

    save_pdf_png(fig, out_dir, "fig5_case_context")


def fig6_prediction_ablation(out_dir: Path, cases: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    fig.suptitle("Forward prediction and geometry ablation across case roles",
                 fontsize=10, fontweight="bold", y=0.98)

    labels = [case_label(x) for x in cases["case_id"]]
    x = np.arange(len(labels))
    width = 0.20
    series = [
        ("Full", "full_mae", COL["blue"]),
        ("Persistence", "persistence_mae", COL["gray"]),
        ("Random edges", "random_edges_mae", COL["orange"]),
        ("No prior", "no_prior_mae", COL["purple"]),
    ]
    for i, (name, col, color) in enumerate(series):
        ax1.bar(x + (i - 1.5) * width, cases[col], width, label=name, color=color, alpha=0.88)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("MAE (standardised response)")
    ax1.set_title("Case-level prediction check", loc="left", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    add_panel_label(ax1, "a")

    deltas = pd.DataFrame({
        "case": labels,
        "Random edges - Full": cases["random_edges_mae"] - cases["full_mae"],
        "No prior - Full": cases["no_prior_mae"] - cases["full_mae"],
        "Persistence - Full": cases["persistence_mae"] - cases["full_mae"],
    })
    y = np.arange(len(labels))
    ax2.axvline(0, color=COL["ink"], linewidth=0.8)
    markers = [("Random edges - Full", COL["orange"], "o"),
               ("No prior - Full", COL["purple"], "s"),
               ("Persistence - Full", COL["gray"], "^")]
    for j, (col, color, marker) in enumerate(markers):
        ax2.scatter(deltas[col], y + (j - 1) * 0.16, color=color, marker=marker, s=35, label=col)
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("MAE difference relative to Full")
    ax2.set_title("Small effects shown as paired deltas", loc="left", fontweight="bold")
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax2.text(0.00015, 2.42, "positive = Full lower MAE", fontsize=6.7, color=COL["muted"])
    add_panel_label(ax2, "b")
    save_pdf_png(fig, out_dir, "fig6_prediction_ablation_summary")


def fig7_sjls_detail(out_dir: Path) -> None:
    metrics = load_json(EXP_DIR / "outputs" / "sjls_dyk1252_411" / "metrics_per_variable.json")
    global_metrics = load_json(EXP_DIR / "outputs" / "sjls_dyk1252_411" / "metrics_global.json")
    ablation = load_json(EXP_DIR / "outputs" / "sjls_dyk1252_411" / "ablation_metrics.json")
    comp = load_component("sjls_dyk1252_411")

    fig = plt.figure(figsize=(10.4, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    fig.suptitle("SJLS: small case-specific geometry-related prediction difference under external TSP",
                 fontsize=10, fontweight="bold", y=0.98)

    models = ["Full Model", "Persistence", "Random Edges", "No Geometry Prior", "No Monitoring"]
    values = [
        ablation["Full Model"]["mae"],
        global_metrics["Persistence"]["mae"],
        ablation["Random Edges"]["mae"],
        ablation["No Geometry Prior"]["mae"],
        ablation["No Monitoring"]["mae"],
    ]
    colors = [COL["blue"], COL["gray"], COL["orange"], COL["purple"], "#B0BEC5"]
    ax1.barh(models, values, color=colors, alpha=0.88)
    ax1.invert_yaxis()
    ax1.set_xlabel("MAE")
    ax1.set_title("Global MAE", loc="left", fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)
    add_panel_label(ax1, "a")

    vars_ = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
    full = np.array([metrics["Full Model"][v]["mae"] for v in vars_])
    pers = np.array([metrics["Persistence"][v]["mae"] for v in vars_])
    ypos = np.arange(len(vars_))
    ax2.axvline(0, color=COL["ink"], linewidth=0.8)
    ax2.barh(ypos, pers - full, color=np.where(pers - full >= 0, COL["blue"], COL["orange"]), alpha=0.85)
    ax2.set_yticks(ypos, vars_)
    ax2.invert_yaxis()
    ax2.set_xlabel("Persistence MAE - Full MAE")
    ax2.set_title("Variable-wise difference", loc="left", fontweight="bold")
    ax2.grid(axis="x", alpha=0.25)
    add_panel_label(ax2, "b")

    torque = comp.drop_duplicates("chainage").sort_values("chainage")
    ax3.plot(torque["chainage"], torque["Torque"], color=COL["blue"], linewidth=1.2, label="Torque")
    ax3.plot(torque["chainage"], torque["ShieldPressure"], color=COL["gray"], linewidth=1.0, label="ShieldPressure")
    ax3.set_xlabel("Chainage (m)")
    ax3.set_ylabel("Standardised response")
    ax3.set_title("Observed SJLS test responses", loc="left", fontweight="bold")
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False, ncol=2)
    add_panel_label(ax3, "c")

    mean_rel = comp.groupby("chainage")["mean_relevance"].mean().reset_index()
    ax4.plot(mean_rel["chainage"], mean_rel["mean_relevance"], color=COL["purple"], linewidth=1.4)
    ax4.set_xlabel("Chainage (m)")
    ax4.set_ylabel("Mean diagnostic relevance")
    ax4.set_title("Relevance used as spatial diagnostic", loc="left", fontweight="bold")
    ax4.grid(alpha=0.25)
    ax4.text(0.03, 0.08, "Diagnostic view, not risk calibration", transform=ax4.transAxes, fontsize=6.8, color=COL["muted"])
    add_panel_label(ax4, "d")
    save_pdf_png(fig, out_dir, "fig7_sjls_prediction_detail")


def fig8_spatial_statistics(out_dir: Path, interp: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10.6, 5.3))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.suptitle("Spatial organisation of learned diagnostic relevance",
                 fontsize=10, fontweight="bold", y=0.98)
    labels = [case_label(x) for x in interp["case_id"]]
    x = np.arange(len(labels))

    ax = axes[0]
    width = 0.34
    ax.bar(x - width / 2, interp["moran_full"], width, color=COL["blue"], label="learned relevance")
    ax.bar(x + width / 2, interp["moran_geometry_only"], width, color=COL["gray"], label="geometry-only")
    ax.axhline(0, color=COL["ink"], linewidth=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial autocorrelation", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=6.8)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.bar(labels, interp["degree_control_r"], color=COL["purple"], alpha=0.88)
    ax.axhline(0, color=COL["ink"], linewidth=0.7)
    ax.set_ylabel("Pearson r")
    ax.set_title("Degree-control check", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "b")

    ax = axes[2]
    colors = [COL["orange"] if v < 0 else COL["teal"] for v in interp["relevance_geology_pearson_r"]]
    ax.bar(labels, interp["relevance_geology_pearson_r"], color=colors, alpha=0.88)
    ax.axhline(0, color=COL["ink"], linewidth=0.7)
    ax.set_ylabel("Pearson r")
    ax.set_title("Relevance--geology association", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "c")

    ax = axes[3]
    ax.bar(labels, interp["strongest_response_abs_pearson"], color=COL["blue"], alpha=0.82)
    for i, name in enumerate(interp["strongest_response_corr"]):
        ax.text(i, interp["strongest_response_abs_pearson"].iloc[i] + 0.025, name,
                ha="center", va="bottom", fontsize=6.8, color=COL["muted"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|Pearson r|")
    ax.set_title("Strongest response association", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "d")
    save_pdf_png(fig, out_dir, "fig8_spatial_statistics")


def fig9_surface_relevance(out_dir: Path) -> None:
    cases = ["bsll_dyk1017_205_h3", "sjls_dyk1252_411"]
    titles = ["BSLL h=3 multi-step interpretation", "SJLS external TSP contrast"]
    fig = plt.figure(figsize=(10.8, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.22], hspace=0.18, wspace=0.20)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cbar_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    fig.suptitle("Component-resolved diagnostic relevance along chainage",
                 fontsize=10, fontweight="bold", y=0.98)

    for idx, (case, title) in enumerate(zip(cases, titles)):
        df = load_component(case)
        pivot = df.pivot(index="component", columns="chainage", values="mean_relevance").loc[COMPONENTS]
        arr = pivot.values
        ax = axes[idx]
        im = ax.imshow(arr, aspect="auto", cmap="cividis", interpolation="nearest")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_yticks(np.arange(len(COMPONENTS)), COMP_LABELS)
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:.0f}" for c in pivot.columns], rotation=0)
        ax.set_xlabel("Chainage (m)")
        ax.set_ylabel("TBM component")
        for y in np.arange(0.5, len(COMPONENTS), 1):
            ax.axhline(y, color="white", linewidth=0.8)
        add_panel_label(ax, chr(ord("a") + idx))
        cax = cbar_axes[idx]
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Mean response-supervised relevance")
    save_pdf_png(fig, out_dir, "fig9_surface_relevance_maps")


def fig10_chainage_evolution(out_dir: Path) -> None:
    df = load_component("bsll_dyk1017_205_h3")
    mean = df.groupby("chainage").agg({
        "mean_relevance": "mean",
        "AdvanceRate": "first",
        "Torque": "first",
        "ShieldPressure": "first",
    }).reset_index()
    comp = df.pivot(index="chainage", columns="component", values="share_relevance")

    fig = plt.figure(figsize=(10.4, 5.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.85, 0.85, 0.85], hspace=0.12)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    fig.suptitle("BSLL h=3 chainage evolution of relevance and monitoring response",
                 fontsize=10, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.plot(mean["chainage"], mean["mean_relevance"], color=COL["purple"], linewidth=1.4)
    ax.set_ylabel("Mean\nrelevance")
    ax.grid(alpha=0.25)
    add_panel_label(ax, "a")

    ax = axes[1]
    bottom = np.zeros(len(comp))
    for component in COMPONENTS:
        vals = comp[component].values
        ax.fill_between(comp.index, bottom, bottom + vals, color=COMP_COLORS[component], alpha=0.85, label=component.replace("_", " "))
        bottom += vals
    ax.set_ylabel("Component\nshare")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=6.4)
    add_panel_label(ax, "b")

    ax = axes[2]
    ax.plot(mean["chainage"], mean["AdvanceRate"], color=COL["blue"], label="AdvanceRate")
    ax.plot(mean["chainage"], mean["Torque"], color=COL["orange"], label="Torque")
    ax.axhline(0, color=COL["grid"], linewidth=0.7)
    ax.set_ylabel("Response\nz-score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=6.8)
    add_panel_label(ax, "c")

    ax = axes[3]
    ax.plot(mean["chainage"], mean["ShieldPressure"], color=COL["gray"], label="ShieldPressure")
    ax.axhline(0, color=COL["grid"], linewidth=0.7)
    ax.set_xlabel("Chainage (m)")
    ax.set_ylabel("Response\nz-score")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.10, "Diagnostic evolution; correlation does not imply causation.",
            transform=ax.transAxes, fontsize=6.8, color=COL["muted"])
    add_panel_label(ax, "d")
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    save_pdf_png(fig, out_dir, "fig10_chainage_evolution")


def make_contact_sheet(out_dir: Path) -> None:
    try:
        from PIL import Image, ImageOps, ImageDraw
    except Exception:
        return
    stems = [
        "fig5_case_context",
        "fig6_prediction_ablation_summary",
        "fig7_sjls_prediction_detail",
        "fig8_spatial_statistics",
        "fig9_surface_relevance_maps",
        "fig10_chainage_evolution",
    ]
    tiles = []
    for stem in stems:
        path = out_dir / f"{stem}.png"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((620, 350))
        canvas = Image.new("RGB", (660, 400), "white")
        canvas.paste(img, ((660 - img.width) // 2, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), stem, fill=(40, 50, 56))
        tiles.append(canvas)
    if not tiles:
        return
    sheet = Image.new("RGB", (1320, 1200), "white")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % 2) * 660, (idx // 2) * 400))
    sheet.save(out_dir / "result_figures_contact_sheet.png")
    ImageOps.grayscale(sheet).save(out_dir / "result_figures_contact_sheet_gray.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result figures from saved evidence.")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    apply_ijgis_style()
    out_dir = Path(args.output_dir)
    cases = load_case_summary()
    interp = load_interp_summary()
    fig5_case_context(out_dir, cases)
    fig6_prediction_ablation(out_dir, cases)
    fig7_sjls_detail(out_dir)
    fig8_spatial_statistics(out_dir, interp)
    fig9_surface_relevance(out_dir)
    fig10_chainage_evolution(out_dir)
    make_contact_sheet(out_dir)
    print(f"Generated result figures in {out_dir}")


if __name__ == "__main__":
    main()
