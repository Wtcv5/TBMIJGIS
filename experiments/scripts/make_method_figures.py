"""Generate manuscript method figures for the GIScience framing.

The figures in this script are method-level schematics. They are intentionally
case-agnostic: BSLL and SJLS appear later in the experimental setting figures.

Usage:
    python scripts/make_method_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.style import IJGIS_COLORS, add_panel_label, apply_ijgis_style


OUT_DEFAULT = Path("outputs/figures")

COLORS = {
    "ink": "#263238",
    "muted": "#607D8B",
    "grid": "#CFD8DC",
    "panel": "#F8FAFB",
    "rock": IJGIS_COLORS["rock"],
    "tbm": IJGIS_COLORS["tbm"],
    "edge": "#8A9AA5",
    "blue": IJGIS_COLORS["full_model"],
    "teal": IJGIS_COLORS["lstm"],
    "orange": IJGIS_COLORS["cutterhead_edge"],
    "purple": IJGIS_COLORS["accent"],
    "gray": "#78909C",
    "light_blue": "#E8F2F8",
    "light_teal": "#E6F3F1",
    "light_orange": "#FFF2E5",
    "light_purple": "#F1EAF6",
}


def save_pdf_png(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axis(ax: plt.Axes, xlim=(0, 1), ylim=(0, 1)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal" if (xlim[1] - xlim[0]) == (ylim[1] - ylim[0]) else "auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def round_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 7.2,
    weight: str = "normal",
    radius: float = 0.025,
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight=weight,
        linespacing=1.15,
    )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["muted"],
    lw: float = 0.9,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_voxel_block(ax: plt.Axes, origin=(0.0, 0.0), scale=1.0) -> None:
    ox, oy = origin
    for ix in range(4):
        for iz in range(3):
            x = ox + scale * (ix * 0.055 + iz * 0.014)
            y = oy + scale * (iz * 0.045)
            rect = patches.Rectangle(
                (x, y),
                scale * 0.045,
                scale * 0.034,
                facecolor=plt.cm.cividis((ix + iz) / 6),
                edgecolor="white",
                linewidth=0.35,
            )
            ax.add_patch(rect)


def draw_tbm_side(ax: plt.Axes, x0=0.0, y0=0.0, scale=1.0, color=COLORS["tbm"]) -> None:
    body = patches.FancyBboxPatch(
        (x0 + 0.055 * scale, y0 + 0.03 * scale),
        0.26 * scale,
        0.09 * scale,
        boxstyle=f"round,pad=0,rounding_size={0.025 * scale}",
        facecolor="#FCE8E2",
        edgecolor=color,
        linewidth=0.9,
    )
    head = patches.Ellipse(
        (x0 + 0.055 * scale, y0 + 0.075 * scale),
        0.07 * scale,
        0.12 * scale,
        facecolor="#F6C7B8",
        edgecolor=color,
        linewidth=0.9,
    )
    ax.add_patch(body)
    ax.add_patch(head)
    for k, label in enumerate(["C", "F", "M", "T"]):
        ax.text(x0 + (0.055 + 0.055 * k) * scale, y0 + 0.142 * scale, label,
                ha="center", va="bottom", fontsize=5.8, color=color)


def draw_graph_nodes(ax: plt.Axes, center=(0.0, 0.0), scale=1.0) -> None:
    cx, cy = center
    rock = np.array([
        [-0.075, 0.045], [-0.035, 0.08], [0.005, 0.04],
        [-0.06, -0.02], [-0.015, 0.0], [0.03, -0.035],
    ])
    tbm = np.array([
        [0.09, 0.07], [0.13, 0.03], [0.14, -0.025], [0.095, -0.065],
    ])
    rock = rock * scale + [cx, cy]
    tbm = tbm * scale + [cx, cy]
    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5), (1, 4)]:
        ax.plot([rock[i, 0], rock[j, 0]], [rock[i, 1], rock[j, 1]],
                color="#B8C5CC", linewidth=0.8)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        ax.plot([tbm[i, 0], tbm[j, 0]], [tbm[i, 1], tbm[j, 1]],
                color="#E3AAA0", linewidth=0.9)
    for i in [1, 2, 4, 5]:
        j = 1 if i < 4 else 2
        ax.plot([rock[i, 0], tbm[j, 0]], [rock[i, 1], tbm[j, 1]],
                color=COLORS["edge"], linewidth=0.65, alpha=0.75)
    ax.scatter(rock[:, 0], rock[:, 1], s=18, c=COLORS["rock"], edgecolors="white", linewidths=0.4, zorder=4)
    ax.scatter(tbm[:, 0], tbm[:, 1], s=22, c=COLORS["tbm"], edgecolors="white", linewidths=0.4, zorder=4)


# ======================================================================
# Fig 1: Framework overview — 3-panel layout matching target figure1.png
# ======================================================================

def _draw_voxel_grid(ax, ox, oy, nx=10, ny=4, nz=3, dx=0.022, dy=0.032, dz=0.02):
    """Draw pseudo-3D TSP voxel grid with depth offset."""
    for ix in range(nx):
        for iz in range(nz):
            x = ox + ix * dx + iz * dz * 0.55
            y = oy + iz * dz * 0.65
            val = (ix + iz * 2) / (nx + nz * 3)
            rect = patches.Rectangle(
                (x, y), dx * 0.88, dy,
                facecolor=plt.cm.Blues(val * 0.55 + 0.25),
                edgecolor="white", linewidth=0.15, alpha=0.82
            )
            ax.add_patch(rect)


def _draw_tbm_cylinder(ax, x0, cy, w, r, show_samples=True):
    """Draw TBM: shield body + cutterhead disc with spokes."""
    # Shield body (rounded rectangle)
    body = patches.FancyBboxPatch(
        (x0, cy - r), w, 2 * r,
        boxstyle=f"round,pad=0,rounding_size={r*0.25}",
        facecolor="#F5EDE8", edgecolor="#A08878", linewidth=0.9, alpha=0.92
    )
    ax.add_patch(body)

    # Cutterhead disc (ellipse)
    head = patches.Ellipse((x0, cy), r * 1.5, 2 * r * 1.05,
                            facecolor="#EDD9CF", edgecolor="#A08878", linewidth=0.9)
    ax.add_patch(head)

    # Spokes on cutterhead
    for angle in np.linspace(0, np.pi, 6)[1:-1]:
        hx = x0 + r * 0.55 * np.cos(angle)
        hy = cy + r * 0.85 * np.sin(angle)
        ax.plot([x0, hx], [cy, hy], color="#B89A86", linewidth=0.45, zorder=2)

    if not show_samples:
        return

    # Cutterhead sample points (orange dots around disc rim)
    n_h = 12
    for a in np.linspace(0, 2 * np.pi, n_h, endpoint=False):
        sx = x0 + r * 0.42 * np.cos(a)
        sy = cy + r * 0.72 * np.sin(a)
        ax.scatter(sx, sy, s=14, c="#D4763C",
                   edgecolors="white", linewidths=0.35, zorder=5)

    # Shield surface samples (top and bottom rows)
    xs_sh = np.linspace(x0 + r * 0.7, x0 + w - r * 0.25, 8)
    for xs in xs_sh[::2]:
        ax.scatter(xs, cy - r * 0.52, s=11, c="#D4763C",
                   edgecolors="white", linewidths=0.3, zorder=5)
        ax.scatter(xs, cy + r * 0.52, s=11, c="#D4763C",
                   edgecolors="white", linewidths=0.3, zorder=5)


def _draw_monitoring_curves(ax, x0, y0, width, height):
    """Draw 4 monitoring response curves: Penetration, Torque, Thrust, Shield pressure."""
    t = np.linspace(0, 1, 80)
    curves = [
        ("Penetration", "#3B7BA8"),
        ("Torque", "#3B9A8C"),
        ("Thrust", "#D4763C"),
        ("Shield pressure", "#7B5AA6"),
    ]
    n = len(curves)
    spacing = height / (n + 1)
    for idx, (label, col) in enumerate(curves):
        yc = y0 + height - spacing * (idx + 1) + spacing * 0.35
        rng = np.random.RandomState(idx * 42 + 7)
        sig = yc + spacing * 0.28 * np.sin(10 * t + idx * 1.4) \
              + spacing * 0.06 * rng.randn(len(t))
        ax.plot(x0 + t * width, sig, color=col, linewidth=0.75)


def _draw_graph_snapshot(ax, cx, cy, scale=1.0, show_labels=False):
    """Draw detailed graph snapshot with rock/TBM nodes and three edge types."""
    s = scale
    # Rock nodes (left cluster, blue squares)
    rock_pts = np.array([
        [-0.09, 0.07], [-0.052, 0.115], [-0.005, 0.07],
        [-0.072, 0.00], [-0.028, 0.01], [0.012, -0.042],
        [-0.105, -0.025], [-0.062, -0.068],
    ]) * s + [cx, cy]
    # TBM nodes (right cluster, orange circles)
    tbm_pts = np.array([
        [0.088, 0.100], [0.128, 0.058], [0.138, 0.002],
        [0.102, -0.048], [0.072, -0.082],
    ]) * s + [cx, cy]

    # E^{rr} edges (solid gray-blue)
    rr_e = [(0,1),(1,2),(3,4),(4,5),(1,4),(0,3),(6,3)]
    for i,j in rr_e:
        ax.plot([rock_pts[i,0],rock_pts[j,0]],[rock_pts[i,1],rock_pts[j,1]],
                color="#90A4AE", linewidth=0.85, zorder=2)
    # E^{mm} edges (solid light brown)
    mm_e = [(0,1),(1,2),(2,3),(3,4)]
    for i,j in mm_e:
        ax.plot([tbm_pts[i,0],tbm_pts[i,0]],[tbm_pts[i,1],tbm_pts[j,1]],
                color="#D4A89A", linewidth=0.95, zorder=2)
    # E^{rm} candidate edges (dashed green)
    rm_e = [(1,0),(2,1),(4,2),(5,3),(1,4)]
    for ri,ti in rm_e:
        ax.plot([rock_pts[ri,0],tbm_pts[ti,0]],[rock_pts[ri,1],tbm_pts[ti,1]],
                color="#5BA86B", linewidth=0.65, linestyle="--", alpha=0.72, zorder=2)

    # Draw nodes
    ax.scatter(rock_pts[:,0], rock_pts[:,1], s=22, c="#4A8EC2",
               marker="s", edgecolors="white", linewidths=0.45, zorder=4)
    ax.scatter(tbm_pts[:,0], tbm_pts[:,1], s=26, c="#D4763C",
               edgecolors="white", linewidths=0.45, zorder=4)

    if show_labels:
        ax.text(cx-0.13*s, cy+0.14*s, "$E^{rr}$", fontsize=6, color="#78909C")
        ax.text(cx+0.11*s, cy-0.12*s, "$E^{mm}$", fontsize=6, color="#B88A7A")


def fig1_framework(output_dir: Path) -> None:
    """Fig 1: Overview of the proposed rock–TBM interaction graph-sequence framework.

    Three panels matching target figure1.png:
      (a) Chainage-referenced rock–TBM representation
      (b) Geometry-constrained interaction graph sequence
      (c) Response-supervised prediction and relevance diagnostics
    """
    fig = plt.figure(figsize=(16.5, 10.0))

    # --- Top title ---
    fig.suptitle(
        "Rock-TBM graph-sequence framework",
        fontsize=13, fontweight="bold", color=COLORS["ink"], y=0.985
    )

    # --- GridSpec: 3 rows ---
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.18, 1.0],
        hspace=0.16, top=0.95, bottom=0.025,
        left=0.015, right=0.985
    )

    # ================================================================
    # Panel (a): Chainage-referenced rock–TBM representation
    # ================================================================
    ax_a = fig.add_subplot(gs[0])
    clean_axis(ax_a, xlim=(-0.005, 1.005), ylim=(-0.02, 1.02))
    add_panel_label(ax_a, "a")
    ax_a.set_title("(a) Chainage-referenced rock–TBM representation",
                   loc="left", fontsize=10, fontweight="bold", pad=3)

    # ---- Left: TSP-derived rock voxels ----
    _draw_voxel_grid(ax_a, ox=0.015, oy=0.20, nx=14, ny=4, nz=3,
                     dx=0.019, dy=0.032, dz=0.017)
    ax_a.text(0.145, 0.93, "TSP-derived rock voxels",
              ha="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])

    # Chainage axis below voxels
    chainages = ["$x_0$", "$x_{t-1}$", "$x_t$", "$x_{t+1}$", "$x_{t+2}$", "..."]
    cx_pos = [0.05, 0.155, 0.26, 0.365, 0.47, 0.56]
    for lbl, xp in zip(chainages, cx_pos):
        ax_a.plot([xp, xp], [0.165, 0.195], color=COLORS["ink"], linewidth=0.65)
        ax_a.text(xp, 0.135, lbl, ha="center", va="top", fontsize=6.5, color=COLORS["ink"])
    ax_a.text(0.305, 0.075, "Chainage / advance direction",
              ha="center", fontsize=6.5, color=COLORS["muted"])
    ax_a.annotate("", xy=(0.54, 0.102), xytext=(0.07, 0.102),
                  arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=0.7))

    # ---- Middle: TBM surface samples ----
    _draw_tbm_cylinder(ax_a, x0=0.32, cy=0.52, w=0.30, r=0.125, show_samples=True)
    ax_a.text(0.48, 0.93, "TBM surface samples",
              ha="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])

    # Arrow from TBM area to monitoring
    arrow(ax_a, (0.64, 0.52), (0.69, 0.52), color=COLORS["muted"], lw=1.2)

    # ---- Right: Monitoring response sequence u_t ----
    _draw_monitoring_curves(ax_a, x0=0.71, y0=0.18, width=0.26, height=0.62)
    ax_a.text(0.84, 0.93, "Monitoring sequence $u_t$",
              ha="center", fontsize=8.0, color=COLORS["ink"], fontweight="bold")

    # ================================================================
    # Panel (b): Geometry-constrained interaction graph sequence
    # ================================================================
    ax_b = fig.add_subplot(gs[1])
    clean_axis(ax_b, xlim=(-0.005, 1.005), ylim=(-0.02, 1.02))
    add_panel_label(ax_b, "b")
    ax_b.set_title("(b) Geometry-constrained interaction graph sequence",
                   loc="left", fontsize=10, fontweight="bold", pad=3)

    # ---- Left sub-section: Graph snapshot at x_t ----
    ax_b.text(0.175, 0.96, "Graph snapshot at $x_t$",
              ha="center", fontsize=8, fontweight="bold", color=COLORS["ink"])

    # Mini TBM cylinder behind graph
    _draw_tbm_cylinder(ax_b, x0=0.055, cy=0.50, w=0.22, r=0.105, show_samples=False)

    # Graph snapshot
    _draw_graph_snapshot(ax_b, cx=0.205, cy=0.52, scale=1.15, show_labels=True)

    # Edge-type legend (bottom-left corner)
    ley = 0.07
    ax_b.plot([0.025, 0.065], [ley, ley], color="#90A4AE", lw=1.3)
    ax_b.text(0.070, ley, "Rock-rock $E^{rr}$", fontsize=5.8, va="center", color="#78909C")
    ax_b.plot([0.175, 0.215], [ley, ley], color="#D4A89A", lw=1.3)
    ax_b.text(0.220, ley, "TBM--TBM $E^{mm}$", fontsize=5.8, va="center", color="#B88A7A")
    ax_b.plot([0.320, 0.360], [ley, ley], color="#5BA86B", lw=1.0, ls="--")
    ax_b.text(0.365, ley, "Rock--TBM $E^{rm}$", fontsize=5.8, va="center", color="#5BA86B")

    # Arrow to middle section
    arrow(ax_b, (0.365, 0.54), (0.405, 0.54), color=COLORS["muted"], lw=1.2)

    # ---- Middle sub-section: Geometry-constrained candidate construction ----
    ax_b.text(0.58, 0.96, "Candidate relations",
              ha="center", fontsize=7.8, fontweight="bold", color=COLORS["ink"])

    # Constraint box background
    cb = patches.FancyBboxPatch(
        (0.405, 0.16), 0.33, 0.70,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="#FAFBFC", edgecolor=COLORS["grid"], linewidth=0.75
    )
    ax_b.add_patch(cb)

    # Active zone diagram (mini circle with voxels and TBM outline)
    az_cx, az_cy = 0.49, 0.58
    az_r = 0.055
    az_circle = patches.Circle((az_cx, az_cy), az_r, facecolor=COLORS["light_blue"],
                                edgecolor=COLORS["blue"], linewidth=0.8)
    ax_b.add_patch(az_circle)
    # mini voxels inside active zone
    for iv in range(4):
        vx = az_cx - az_r*0.55 + iv * az_r*0.38
        vy = az_cy - az_r*0.25 + (iv % 2) * az_r*0.38
        vrect = patches.Rectangle((vx, vy), az_r*0.32, az_r*0.28,
                                   facecolor=plt.cm.Blues(0.5),
                                   edgecolor="white", linewidth=0.15, alpha=0.8)
        ax_b.add_patch(vrect)
    # mini TBM arc inside active zone
    arc_theta = np.linspace(-np.pi/3, np.pi/3, 30)
    ax_b.plot(az_cx + az_r*0.72*np.cos(arc_theta),
              az_cy + az_r*0.68*np.sin(arc_theta),
              color=COLORS["tbm"], linewidth=1.0)
    ax_b.text(az_cx, az_cy - az_r - 0.035, "active\nzone",
              ha="center", va="top", fontsize=5.5, color=COLORS["blue"])

    # Distance threshold
    dt_x1, dt_x2 = 0.54, 0.605
    dt_y = 0.60
    ax_b.annotate("", xy=(dt_x2, dt_y), xytext=(dt_x1, dt_y),
                  arrowprops=dict(arrowstyle="<->", color=COLORS["ink"], lw=0.85))
    ax_b.text((dt_x1+dt_x2)/2, dt_y+0.055, "distance",
              ha="center", fontsize=5.3, color=COLORS["ink"])
    ax_b.text((dt_x1+dt_x2)/2, dt_y-0.055, "$d \\leq d_e$",
              ha="center", fontsize=5.8, color=COLORS["ink"])

    # Normal compatibility
    nc_x = 0.66
    nc_y = 0.58
    ax_b.scatter([nc_x], [nc_y], s=28, c=COLORS["tbm"],
                 edgecolors="white", linewidths=0.4, zorder=5)
    ax_b.scatter([nc_x - 0.055], [nc_y + 0.055], s=22, c=COLORS["rock"],
                 marker="s", edgecolors="white", linewidths=0.35, zorder=5)
    ax_b.arrow(nc_x, nc_y, 0.045, 0.045, width=0.0016,
               head_width=0.014, head_length=0.012, color=COLORS["tbm"], zorder=4)
    ax_b.annotate("", xy=(nc_x - 0.015, nc_y + 0.015),
                  xytext=(nc_x - 0.055, nc_y + 0.055),
                  arrowprops=dict(arrowstyle="->", color=COLORS["rock"], lw=0.8))
    ax_b.text(nc_x + 0.035, nc_y + 0.072, "$n_j(t)$",
              ha="center", fontsize=5.6, color=COLORS["tbm"])
    ax_b.text(nc_x - 0.060, nc_y + 0.085, "$c_i-p_j(t)$",
              ha="center", fontsize=5.4, color=COLORS["rock"])
    ax_b.text(nc_x + 0.005, nc_y - 0.095,
              "$\\kappa=\\max(0, n_j^T(c_i-p_j)/(d+\\epsilon))$",
              ha="center", fontsize=5.0, color=COLORS["ink"])

    # Arrow to right section
    arrow(ax_b, (0.745, 0.54), (0.780, 0.54), color=COLORS["muted"], lw=1.2)

    # ---- Right sub-section: Graph sequence along chainage ----
    ax_b.text(0.89, 0.96, "Graph sequence along chainage",
              ha="center", fontsize=8, fontweight="bold", color=COLORS["ink"])

    # Sequence of graph snapshots
    seq_info = [
        (0.81, rf"$G_{{t-K+1}}$", 0.52),
        (0.87, rf"$G_{{t-K+2}}$", 0.62),
        (None, "...", None),
        (0.94, "$G_t$", 0.76),
    ]
    for i, (sx, lbl, sc) in enumerate(seq_info):
        if i == 2:
            ax_b.text(0.91, 0.53, lbl, ha="center", va="center",
                      fontsize=12, color=COLORS["muted"])
            continue
        bw, bh = 0.058, 0.36
        round_box(ax_b, (sx-bw/2, 0.34), bw, bh, lbl,
                  "#FFFFFF", COLORS["purple"], fontsize=6.5, radius=0.010)
        if sc is not None:
            _draw_graph_snapshot(ax_b, sx, 0.52, scale=sc*0.42, show_labels=False)
        if i < 2:
            arrow(ax_b, (sx+bw/2+0.004, 0.52),
                  (seq_info[i+1][0]-0.032 if seq_info[i+1][0] else 0.94, 0.52),
                  color=COLORS["purple"], lw=0.8)

    # Time / advance arrow
    ax_b.annotate("", xy=(0.97, 0.14), xytext=(0.79, 0.14),
                  arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=1.0))
    ax_b.text(0.88, 0.07, "advance", ha="center",
              fontsize=6.5, color=COLORS["muted"])

    # Bottom legend bar
    bleg_items = [
        ("Rock voxels", "#4A8EC2", "s"),
        ("TBM surface nodes", "#D4763C", "o"),
        ("Rock adjacency", "#90A4AE", "-"),
        ("TBM adjacency", "#D4A89A", "-"),
        ("Candidate edge", "#5BA86B", "--"),
    ]
    blx = 0.018
    for idx, (lbl, clr, mk) in enumerate(bleg_items):
        lx = blx + idx * 0.198
        if mk == "s":
            ax_b.scatter(lx, 0.015, s=18, c=clr, marker="s",
                         edgecolors="white", linewidths=0.35)
        elif mk == "o":
            ax_b.scatter(lx, 0.015, s=18, c=clr,
                         edgecolors="white", linewidths=0.35)
        else:
            ax_b.plot([lx, lx+0.018], [0.015, 0.015], color=clr,
                      lw=1.2, linestyle="-" if mk == "-" else "--")
        ax_b.text(lx+0.023, 0.015, lbl, fontsize=4.6, va="center",
                  color=COLORS["muted"], linespacing=1.1)

    # ================================================================
    # Panel (c): Response-supervised prediction and relevance diagnostics
    # ================================================================
    ax_c = fig.add_subplot(gs[2])
    clean_axis(ax_c, xlim=(-0.005, 1.005), ylim=(-0.02, 1.02))
    add_panel_label(ax_c, "c")
    ax_c.set_title("(c) Prediction and diagnostic relevance",
                   loc="left", fontsize=10, fontweight="bold", pad=3)

    # Vertical divider between left (model pipeline) and right (evaluation outputs)
    div_x = 0.61
    ax_c.axvline(x=div_x, ymin=0.03, ymax=0.92,
                 color=COLORS["grid"], linewidth=0.85, ls="-")

    # ---- Left half: GNN-GRU model pipeline ----
    ax_c.text(0.175, 0.94, "Inputs", fontsize=9, fontweight="bold", color=COLORS["ink"])

    # Input: Graph sequence boxes (3 small boxes)
    gx_positions = [0.018, 0.062, 0.106]
    for i, gx in enumerate(gx_positions):
        round_box(ax_c, (gx, 0.70), 0.040, 0.095, rf"$G_{{t-K+{i}}}$",
                  "#FFFFFF", COLORS["purple"], fontsize=5.5, radius=0.009)
    ax_c.text(0.072, 0.815, "Graph sequence\n$G_{t-K+1:t}$",
              ha="center", fontsize=6.2, color=COLORS["purple"])

    # Input: Monitoring sequence box
    round_box(ax_c, (0.018, 0.46), 0.138, 0.135,
              "$u_{t-K+1:t}$\nmonitoring sequence",
              COLORS["light_teal"], COLORS["teal"], fontsize=6.2, weight="bold")

    # Model box (GNN-GRU)
    round_box(ax_c, (0.185, 0.51), 0.115, 0.29,
              "GNN-GRU\nmodel", COLORS["light_orange"],
              COLORS["orange"], fontsize=8.5, weight="bold", radius=0.018)

    # Arrows into model
    arrow(ax_c, (0.156, 0.75), (0.185, 0.69), color=COLORS["purple"], lw=0.9)
    arrow(ax_c, (0.156, 0.53), (0.185, 0.62), color=COLORS["teal"], lw=0.9)

    # Arrow from model to output
    arrow(ax_c, (0.300, 0.655), (0.340, 0.655), color=COLORS["muted"], lw=1.1)

    # Forward response prediction panel
    pred_bg = patches.FancyBboxPatch(
        (0.350, 0.43), 0.245, 0.45,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#FEFEFE", edgecolor=COLORS["grid"], linewidth=0.65
    )
    ax_c.add_patch(pred_bg)
    ax_c.text(0.4725, 0.865, "Forward response prediction",
              ha="center", fontsize=7.5, fontweight="bold", color=COLORS["ink"])

    # Simulated observed vs predicted curves
    tp = np.linspace(0, 1, 55)
    xp_curve = 0.368 + tp * 0.21
    x_div = 0.4725  # divider position within prediction box
    p_colors = ["#3B9A8C", "#D4763C", "#3B7BA8", "#7B5AA6"]
    p_labels = ["Penetration", "Torque", "Thrust", "Shield pressure"]
    sp_p = 0.080
    for idx, (col, lbl) in enumerate(zip(p_colors, p_labels)):
        yc = 0.80 - idx * sp_p
        rng = np.random.RandomState(idx * 137 + 3)
        obs = yc + 0.026*np.sin(7*tp + idx*1.1) + 0.007*rng.randn(len(tp))
        prd = obs + 0.006*rng.randn(len(tp))
        split = int(len(tp)*0.54)
        ax_c.plot(xp_curve[:split], obs[:split], color=col, lw=0.8)
        ax_c.plot(xp_curve[split:], prd[split:], color=col, lw=0.8, ls="--")
        ax_c.text(0.358, yc+0.018, lbl, fontsize=5.0, color=col)

    # Divider line inside prediction plot
    ax_c.axvline(x=x_div, ymin=0.44, ymax=0.83,
                 color=COLORS["grid"], linewidth=0.55, ls="-")
    ax_c.text(0.450, 0.435, "Observed\n(up to $x_t$)",
              ha="center", fontsize=5.0, color=COLORS["muted"])
    ax_c.text(0.500, 0.435, "Predicted\n(from $x_t$ to $x_{t+h}$)",
              ha="center", fontsize=5.0, color=COLORS["muted"])
    # Chainage axis under prediction
    ax_c.annotate("", xy=(0.588, 0.445), xytext=(0.368, 0.445),
                  arrowprops=dict(arrowstyle="<-", color=COLORS["ink"], lw=0.65))
    ax_c.text(0.398, 0.420, "$x_{t-K+1}$", fontsize=5.5, ha="center")
    ax_c.text(0.4725, 0.420, "$x_t$", fontsize=5.5, ha="center")
    ax_c.text(0.560, 0.420, "$x_{t+h}$", fontsize=5.5, ha="center")
    ax_c.text(0.4725, 0.395, "Chainage $x$", fontsize=5.8, ha="center",
              color=COLORS["ink"])

    # ---- Right half: Evaluation outputs ----

    # Right-top: ablation variants without schematic performance values
    ax_c.text(0.815, 0.94, "Ablations",
              ha="center", fontsize=7.5, fontweight="bold", color=COLORS["ink"])

    bar_bg = patches.FancyBboxPatch(
        (0.630, 0.54), 0.345, 0.34,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor="#FEFEFE", edgecolor=COLORS["grid"], linewidth=0.65
    )
    ax_c.add_patch(bar_bg)

    variants = [
        ("Full", "screened\nrelations", "#3B7BA8"),
        ("Random\nedges", "topology\ncontrol", "#D4763C"),
        ("No prior", "remove\n$\\pi^{rm}_{ij}$", "#7B5AA6"),
        ("No\nmonitoring", "remove\nmonitoring sequence", "#999999"),
    ]
    for idx, (lbl, desc, col) in enumerate(variants):
        bx = 0.650 + idx * 0.078
        ax_c.add_patch(patches.FancyBboxPatch(
            (bx, 0.645), 0.063, 0.125,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor=col, edgecolor="white", lw=0.45, alpha=0.88
        ))
        ax_c.text(bx + 0.0315, 0.705, lbl, ha="center", va="center",
                  fontsize=5.4, color="white", fontweight="bold")
        ax_c.text(bx + 0.0315, 0.610, desc, ha="center", va="top",
                  fontsize=5.0, color=COLORS["ink"])

    # Right-bottom: Component-resolved relevance heatmap
    ax_c.text(0.815, 0.512,
              "Component-chainage diagnostic relevance",
              ha="center", fontsize=6.5, fontweight="bold", color=COLORS["ink"])

    hm_bg = patches.FancyBboxPatch(
        (0.630, 0.055), 0.345, 0.395,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor="#FFFFFF", edgecolor=COLORS["grid"], linewidth=0.65
    )
    ax_c.add_patch(hm_bg)

    # Heatmap grid
    hm_nc = 24   # chainage columns
    hm_nr = 6    # component rows
    hm_palette = [
        "#FFF5EB", "#FFE4CC", "#FFD1A6", "#FFBC7D",
        "#FA8C5A", "#E85D30", "#C23B18"
    ]
    for ic in range(hm_nr):
        for ir in range(hm_nc):
            base = 0.25 + 0.55*np.exp(-((ir - hm_nc*0.68)**2)/(hm_nc**2*0.07))
            noise = np.random.RandomState(ic*11+ir).randn()*0.09
            val = np.clip(base + noise + ic*0.04, 0, 1)
            ci = min(int(val*(len(hm_palette)-1)), len(hm_palette)-1)
            cw = 0.330/hm_nc*0.94
            ch = 0.360/hm_nr*0.88
            cx_hm = 0.642 + ir*(0.330/hm_nc)
            cy_hm = 0.072 + ic*(0.360/hm_nr)
            ax_c.add_patch(patches.Rectangle(
                (cx_hm, cy_hm), cw, ch,
                facecolor=hm_palette[ci], edgecolor="none"
            ))

    # Heatmap labels
    ax_c.text(0.642, 0.438, "Cutterhead", fontsize=5.5, va="bottom",
              color=COLORS["tbm"], fontweight="bold")
    ax_c.text(0.642, 0.070, "Shield", fontsize=5.5, va="top",
              color=COLORS["tbm"], fontweight="bold")
    ax_c.text(0.810, 0.042, "$x_{t-K+1}$", fontsize=5.5, ha="center")
    ax_c.text(0.978, 0.042, "$x_{t+h}$", fontsize=5.5, ha="center")
    ax_c.text(0.900, 0.022, "Chainage $x$", fontsize=5.8, ha="center",
              color=COLORS["ink"])

    save_pdf_png(fig, output_dir, "fig1_method_framework")


def fig2_spatial_entities(output_dir: Path) -> None:
    fig = plt.figure(figsize=(10.2, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.1, 1.0], height_ratios=[1, 1], wspace=0.22, hspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for ax in axes:
        clean_axis(ax)

    # a: coordinate and chainage support
    ax = axes[0]
    ax.set_title("Chainage-referenced tunnel space", loc="left", fontweight="bold")
    ax.plot([0.08, 0.92], [0.34, 0.34], color=COLORS["ink"], linewidth=1.0)
    for x in np.linspace(0.12, 0.88, 7):
        ax.plot([x, x], [0.315, 0.365], color=COLORS["ink"], linewidth=0.6)
    ax.text(0.5, 0.23, "chainage x", ha="center", fontsize=7.0)
    ax.add_patch(patches.Rectangle((0.28, 0.42), 0.35, 0.12, facecolor=COLORS["light_blue"], edgecolor=COLORS["rock"], linewidth=0.8))
    ax.text(0.455, 0.48, "TSP support", ha="center", va="center", fontsize=7.0)
    ax.add_patch(patches.Rectangle((0.55, 0.12), 0.18, 0.44, facecolor="#F9F9F9", edgecolor=COLORS["tbm"], linewidth=0.9))
    ax.text(0.64, 0.60, "TBM window", ha="center", fontsize=6.8, color=COLORS["tbm"])
    arrow(ax, (0.72, 0.34), (0.87, 0.34), color=COLORS["orange"])
    ax.text(0.82, 0.39, "advance", fontsize=6.8, color=COLORS["orange"])
    add_panel_label(ax, "a")

    # b: voxel field
    ax = axes[1]
    ax.set_title("Rock entity set $V^r_t$", loc="left", fontweight="bold")
    for ix in range(7):
        for iy in range(5):
            val = (ix + 0.55 * iy) / 9.8
            ax.add_patch(patches.Rectangle((0.12 + ix * 0.08, 0.18 + iy * 0.085), 0.07, 0.072,
                                           facecolor=plt.cm.cividis(val), edgecolor="white", linewidth=0.45))
    ax.text(0.40, 0.10, "voxel attributes $g_i$: P-wave, S-wave, lithology proxy", ha="center", fontsize=6.5)
    ax.text(0.73, 0.66, "$c_i=(x,y,z)$", fontsize=7.0, color=COLORS["rock"])
    add_panel_label(ax, "b")

    # c: TBM surface
    ax = axes[2]
    ax.set_title("Machine surface entity set $V^m_t$", loc="left", fontweight="bold")
    draw_tbm_side(ax, x0=0.13, y0=0.35, scale=1.9)
    xs = np.linspace(0.23, 0.72, 9)
    ys = 0.50 + 0.10 * np.sin(np.linspace(0, 2 * np.pi, 9))
    ax.scatter(xs, ys, s=18, c=COLORS["tbm"], edgecolors="white", linewidths=0.5)
    for x, yv in zip(xs[::2], ys[::2]):
        ax.arrow(x, yv, 0.0, 0.08, width=0.002, head_width=0.018, head_length=0.025,
                 color=COLORS["tbm"], length_includes_head=True)
    ax.text(0.50, 0.18, "surface node $p_j(t)$ with normal $n_j(t)$ and component label", ha="center", fontsize=6.5)
    add_panel_label(ax, "c")

    # d: monitoring sequence
    ax = axes[3]
    ax.set_title("Monitoring response sequence", loc="left", fontweight="bold")
    t = np.linspace(0.12, 0.92, 80)
    for k, (color, offset) in enumerate([(COLORS["blue"], 0.55), (COLORS["teal"], 0.42), (COLORS["orange"], 0.29)]):
        ax.plot(t, offset + 0.035 * np.sin(10 * t + k), color=color, linewidth=1.0)
    ax.add_patch(patches.Rectangle((0.58, 0.18), 0.20, 0.48, facecolor="#FFFFFF", edgecolor=COLORS["purple"], linewidth=0.9, linestyle="--"))
    ax.text(0.68, 0.70, "history window K", ha="center", fontsize=6.8, color=COLORS["purple"])
    ax.text(0.50, 0.10, "$u_{t-K+1:t}$ aligned with graph snapshots", ha="center", fontsize=6.8)
    add_panel_label(ax, "d")

    # e: heterogeneous graph
    ax = axes[4]
    ax.set_title("Heterogeneous graph snapshot $G_t$", loc="left", fontweight="bold")
    draw_graph_nodes(ax, center=(0.48, 0.47), scale=1.8)
    ax.text(0.25, 0.18, "$E^{rr}_t$", color=COLORS["rock"], fontsize=7.0)
    ax.text(0.72, 0.23, "$E^{mm}_t$", color=COLORS["tbm"], fontsize=7.0)
    ax.text(0.50, 0.72, "$E^{rm}_t$", color=COLORS["edge"], fontsize=7.0)
    add_panel_label(ax, "e")

    # f: supervised target
    ax = axes[5]
    ax.set_title("Prediction target and interpretation", loc="left", fontweight="bold")
    round_box(ax, (0.08, 0.60), 0.32, 0.14, "$G_{t-K+1:t}$", "#FFFFFF", COLORS["purple"], fontsize=8)
    round_box(ax, (0.08, 0.35), 0.32, 0.14, "$u_{t-K+1:t}$", "#FFFFFF", COLORS["teal"], fontsize=8)
    round_box(ax, (0.60, 0.49), 0.28, 0.14, "$\\hat{r}_{t+h}$", COLORS["light_blue"], COLORS["blue"], fontsize=8, weight="bold")
    arrow(ax, (0.41, 0.67), (0.59, 0.57), color=COLORS["purple"])
    arrow(ax, (0.41, 0.42), (0.59, 0.55), color=COLORS["teal"])
    ax.scatter([0.28, 0.33, 0.36], [0.20, 0.24, 0.18], s=[26, 44, 30], c=plt.cm.cividis([0.35, 0.78, 0.55]), edgecolors="white")
    ax.text(0.50, 0.22, "learned relevance aggregated by component and chainage", ha="center", fontsize=6.6)
    add_panel_label(ax, "f")

    fig.suptitle("Spatial entity formalisation before case-specific experiments",
                 fontsize=10, fontweight="bold", y=0.98)
    save_pdf_png(fig, output_dir, "fig2_spatial_entity_formalisation")


def fig3_geometry_constraints(output_dir: Path) -> None:
    fig = plt.figure(figsize=(10.4, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1, 1], wspace=0.22, hspace=0.30)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax in axes:
        clean_axis(ax)

    # a longitudinal active zones
    ax = axes[0]
    ax.set_title("Longitudinal active-zone screening", loc="left", fontweight="bold")
    ax.add_patch(patches.Rectangle((0.07, 0.24), 0.86, 0.34, facecolor="#ECEFF1", edgecolor="none"))
    ax.add_patch(patches.Rectangle((0.20, 0.24), 0.27, 0.34, facecolor=COLORS["light_orange"], edgecolor="none"))
    ax.add_patch(patches.Rectangle((0.47, 0.24), 0.32, 0.34, facecolor=COLORS["light_blue"], edgecolor="none"))
    draw_tbm_side(ax, x0=0.455, y0=0.305, scale=1.15)
    ax.axvline(0.47, color=COLORS["orange"], linewidth=0.9)
    ax.text(0.47, 0.64, "$x_{face}(t)$", ha="center", fontsize=7.0, color=COLORS["orange"])
    ax.annotate("", xy=(0.20, 0.17), xytext=(0.47, 0.17), arrowprops=dict(arrowstyle="<->", color=COLORS["orange"], lw=0.8))
    ax.text(0.335, 0.10, "cutterhead look-ahead $L_f$", ha="center", fontsize=6.6, color=COLORS["orange"])
    ax.annotate("", xy=(0.47, 0.76), xytext=(0.79, 0.76), arrowprops=dict(arrowstyle="<->", color=COLORS["blue"], lw=0.8))
    ax.text(0.63, 0.82, "shield active zone $\\tau_{\\mathrm{zone}}$", ha="center", fontsize=6.6, color=COLORS["blue"])
    add_panel_label(ax, "a")

    # b cross section normal/distance
    ax = axes[1]
    ax.set_title("Distance and normal compatibility", loc="left", fontweight="bold")
    center = (0.50, 0.48)
    ax.add_patch(patches.Circle(center, 0.25, facecolor="#FFFFFF", edgecolor=COLORS["tbm"], linewidth=1.0))
    ax.add_patch(patches.Circle(center, 0.34, facecolor="none", edgecolor=COLORS["grid"], linewidth=0.8, linestyle="--"))
    tbm_points = np.array([[0.33, 0.62], [0.61, 0.70], [0.72, 0.40], [0.42, 0.24]])
    rock_points = np.array([[0.24, 0.78], [0.62, 0.88], [0.83, 0.36], [0.28, 0.24], [0.74, 0.76]])
    ax.scatter(tbm_points[:, 0], tbm_points[:, 1], s=28, c=COLORS["tbm"], edgecolors="white", zorder=4)
    ax.scatter(rock_points[:, 0], rock_points[:, 1], s=25, c=COLORS["rock"], edgecolors="white", zorder=4)
    for rp, tp, ok in [(rock_points[0], tbm_points[0], True), (rock_points[1], tbm_points[1], True),
                       (rock_points[2], tbm_points[2], True), (rock_points[3], tbm_points[3], False),
                       (rock_points[4], tbm_points[1], False)]:
        ax.plot([rp[0], tp[0]], [rp[1], tp[1]], color=COLORS["orange"] if ok else "#B0BEC5",
                linewidth=1.0 if ok else 0.7, linestyle="-" if ok else "--", alpha=0.9)
    ax.arrow(0.61, 0.70, 0.07, 0.07, width=0.003, head_width=0.02, head_length=0.025, color=COLORS["tbm"])
    ax.text(0.70, 0.78, "$n_j(t)$", fontsize=7.0, color=COLORS["tbm"])
    ax.text(0.48, 0.10, "$d_{ij}\\leq\\tau_{\\mathrm{edge}}$,  $\\kappa_{ij}=\\max(0,n_j^T(c_i-p_j)/d_{ij})$", ha="center", fontsize=6.7)
    add_panel_label(ax, "b")

    # c candidate to retained edges
    ax = axes[2]
    ax.set_title("Candidate relation filtering", loc="left", fontweight="bold")
    round_box(ax, (0.08, 0.64), 0.24, 0.13, "active\nrock voxels", COLORS["light_blue"], COLORS["rock"])
    round_box(ax, (0.08, 0.34), 0.24, 0.13, "TBM surface\npatches", "#FCE8E2", COLORS["tbm"])
    round_box(ax, (0.40, 0.50), 0.22, 0.14, "candidate\npairs", "#FFFFFF", COLORS["gray"])
    round_box(ax, (0.70, 0.50), 0.22, 0.14, "plausible\n$E^{rm}_t$", COLORS["light_orange"], COLORS["orange"], weight="bold")
    arrow(ax, (0.32, 0.70), (0.40, 0.58))
    arrow(ax, (0.32, 0.40), (0.40, 0.56))
    arrow(ax, (0.62, 0.57), (0.70, 0.57), color=COLORS["orange"])
    ax.text(0.66, 0.64, "$\\pi^{rm}_{ij}=\\exp(-d_{ij}/\\tau_{\\mathrm{edge}})\\kappa_{ij}$", ha="center", fontsize=6.6, color=COLORS["orange"])
    ax.text(0.50, 0.25, "screen by active zone, distance, normal alignment, and geometry prior", ha="center", fontsize=6.7)
    add_panel_label(ax, "c")

    # d ablation meaning
    ax = axes[3]
    ax.set_title("What the geometry ablations test", loc="left", fontweight="bold")
    rows = [
        ("Full", "screened edges + prior", COLORS["blue"]),
        ("Random edges", "same edge count, randomized endpoints", COLORS["orange"]),
        ("No prior", "screened edges, remove prior bias", COLORS["purple"]),
        ("No monitoring", "remove monitoring sequence input", COLORS["gray"]),
    ]
    for idx, (name, desc, color) in enumerate(rows):
        y = 0.76 - idx * 0.18
        ax.add_patch(patches.Circle((0.12, y), 0.030, facecolor=color, edgecolor="white", linewidth=0.5))
        ax.text(0.19, y + 0.018, name, ha="left", va="center", fontsize=7.2, fontweight="bold", color=COLORS["ink"])
        ax.text(0.19, y - 0.040, desc, ha="left", va="center", fontsize=6.6, color=COLORS["muted"])
    ax.text(0.50, 0.08, "Ablations isolate relation plausibility and monitoring dependence, not contact recovery.", ha="center", fontsize=6.7)
    add_panel_label(ax, "d")

    fig.suptitle("Geometry-compatible rock-machine relation construction",
                 fontsize=10, fontweight="bold", y=0.98)
    save_pdf_png(fig, output_dir, "fig3_geometry_constrained_edges")


def fig4_network_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    clean_axis(ax)
    ax.text(0.50, 0.94, "Graph-sequence network for response-supervised spatial learning",
            ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["ink"])

    # graph sequence inputs
    gx = [0.06, 0.12, 0.18, 0.24]
    for i, x in enumerate(gx):
        round_box(ax, (x, 0.67 - 0.018 * i), 0.105, 0.085, rf"$G_{{t-{3-i}}}$",
                  "#FFFFFF", COLORS["purple"], fontsize=7.5, radius=0.012)
        draw_graph_nodes(ax, center=(x + 0.052, 0.70 - 0.018 * i), scale=0.28)
    ax.text(0.15, 0.82, "heterogeneous graph sequence", ha="center", fontsize=7.0, color=COLORS["purple"])

    # monitoring input
    round_box(ax, (0.07, 0.33), 0.20, 0.12, "$u_{t-K+1:t}$\nmonitoring sequence",
              COLORS["light_teal"], COLORS["teal"], fontsize=7.2, weight="bold")
    for k in range(4):
        ax.plot([0.09, 0.25], [0.28 - 0.028 * k, 0.28 - 0.028 * k],
                color=[COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["gray"]][k], linewidth=1.0)

    # encoders
    round_box(ax, (0.36, 0.69), 0.18, 0.12, "type-specific\nnode encoders",
              COLORS["light_blue"], COLORS["blue"], fontsize=7.2, weight="bold")
    round_box(ax, (0.36, 0.52), 0.18, 0.12, "edge feature\nencoder",
              "#FFFFFF", COLORS["orange"], fontsize=7.2, weight="bold")
    round_box(ax, (0.58, 0.60), 0.20, 0.16, "rock-machine\nattention message passing\n$s_{ij}=MLP+\\beta\\pi^{rm}_{ij}$",
              COLORS["light_orange"], COLORS["orange"], fontsize=6.8, weight="bold")
    round_box(ax, (0.58, 0.39), 0.20, 0.12, "graph readout\n$z_t$",
              "#FFFFFF", COLORS["purple"], fontsize=7.2, weight="bold")
    round_box(ax, (0.36, 0.31), 0.18, 0.12, "homogeneous\n$E^{rr}, E^{mm}$ propagation",
              "#F4F7F8", COLORS["gray"], fontsize=6.8)

    for y in [0.73, 0.57, 0.37]:
        arrow(ax, (0.30, 0.70 if y > 0.6 else 0.39), (0.36, y), color=COLORS["muted"])
    arrow(ax, (0.54, 0.75), (0.58, 0.69), color=COLORS["blue"])
    arrow(ax, (0.54, 0.58), (0.58, 0.66), color=COLORS["orange"])
    arrow(ax, (0.54, 0.37), (0.58, 0.63), color=COLORS["gray"], rad=0.08)
    arrow(ax, (0.68, 0.60), (0.68, 0.51), color=COLORS["purple"])

    # temporal part
    round_box(ax, (0.36, 0.12), 0.18, 0.12, "concatenate\n$[z_t || u_t]$",
              "#FFFFFF", COLORS["teal"], fontsize=7.2, weight="bold")
    round_box(ax, (0.62, 0.15), 0.16, 0.14, "GRU temporal\nencoder",
              COLORS["light_purple"], COLORS["purple"], fontsize=7.2, weight="bold")
    round_box(ax, (0.84, 0.48), 0.12, 0.12, "response\nhead",
              COLORS["light_blue"], COLORS["blue"], fontsize=7.2, weight="bold")
    round_box(ax, (0.84, 0.26), 0.12, 0.12, "diagnostic\naggregation",
              "#EEF3F6", COLORS["gray"], fontsize=7.0, weight="bold")
    ax.text(0.90, 0.65, "$\\hat{r}_{t+h}$", ha="center", fontsize=8.2, color=COLORS["blue"], fontweight="bold")
    ax.scatter([0.875, 0.905, 0.925], [0.19, 0.22, 0.18], s=[28, 45, 30],
               c=plt.cm.cividis([0.30, 0.78, 0.55]), edgecolors="white", linewidths=0.5)
    ax.text(0.90, 0.11, "diagnostic relevance", ha="center", fontsize=6.8, color=COLORS["muted"])

    arrow(ax, (0.68, 0.39), (0.45, 0.24), color=COLORS["purple"], rad=-0.15)
    arrow(ax, (0.27, 0.39), (0.36, 0.18), color=COLORS["teal"], rad=-0.05)
    arrow(ax, (0.54, 0.18), (0.62, 0.22), color=COLORS["teal"])
    arrow(ax, (0.78, 0.22), (0.84, 0.51), color=COLORS["blue"], rad=0.12)
    arrow(ax, (0.78, 0.22), (0.84, 0.32), color=COLORS["gray"], rad=-0.08)

    # attention callout
    callout_path = MplPath(
        [(0.80, 0.74), (0.95, 0.74), (0.95, 0.86), (0.80, 0.86), (0.80, 0.74)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY],
    )
    ax.add_patch(patches.PathPatch(callout_path, facecolor="#FFFFFF", edgecolor=COLORS["grid"], linewidth=0.8))
    ax.text(0.875, 0.80, "Attention scores are\nmodel relevance,\nnot contact force.",
            ha="center", va="center", fontsize=6.5, color=COLORS["muted"], linespacing=1.1)

    save_pdf_png(fig, output_dir, "fig4_graph_sequence_network")


def make_contact_sheet(output_dir: Path) -> None:
    try:
        from PIL import Image, ImageOps, ImageDraw
    except Exception:
        return

    pngs = [
        output_dir / "fig1_method_framework.png",
        output_dir / "fig2_spatial_entity_formalisation.png",
        output_dir / "fig3_geometry_constrained_edges.png",
        output_dir / "fig4_graph_sequence_network.png",
    ]
    images = []
    for path in pngs:
        if path.exists():
            img = Image.open(path).convert("RGB")
            img.thumbnail((760, 430))
            canvas = Image.new("RGB", (800, 480), "white")
            canvas.paste(img, ((800 - img.width) // 2, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 12), path.stem, fill=(40, 50, 56))
            images.append(canvas)
    if not images:
        return
    sheet = Image.new("RGB", (1600, 960), "white")
    for idx, img in enumerate(images):
        sheet.paste(img, ((idx % 2) * 800, (idx // 2) * 480))
    sheet.save(output_dir / "method_figures_contact_sheet.png")
    gray = ImageOps.grayscale(sheet)
    gray.save(output_dir / "method_figures_contact_sheet_gray.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate method figures for the TBM graph-sequence manuscript.")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT), help="Output directory relative to experiments/.")
    args = parser.parse_args()

    apply_ijgis_style()
    output_dir = Path(args.output_dir)
    fig1_framework(output_dir)
    fig2_spatial_entities(output_dir)
    fig3_geometry_constraints(output_dir)
    fig4_network_architecture(output_dir)
    make_contact_sheet(output_dir)
    print(f"Generated method figures in {output_dir}")


if __name__ == "__main__":
    main()
