"""Publication-quality graph construction visualizations."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpecFromSubplotSpec

from src.visualization.style import PALETTE, add_panel_label, apply_publication_style, figure_size, save_figure

os.environ.setdefault("PYVISTA_OFF_SCREEN", "True")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

COMPONENT_COLORS: dict[int, str] = {
    0: "#D73027",
    1: "#FC8D59",
    2: "#FEE08B",
    3: "#91BFDB",
}
COMPONENT_NAMES = {0: "Cutterhead", 1: "Front shield", 2: "Middle shield", 3: "Tail shield"}

MAX_ROCK_POINTS = 3000
MAX_EDGE_LINES = 500
PV_WINDOW_SIZE = (1400, 1000)


# ── Helpers ──

def _prepare_tbm_components(
    tbm_components: Optional[np.ndarray],
    n_tbm_points: int,
) -> np.ndarray:
    if tbm_components is None:
        return np.zeros(n_tbm_points, dtype=np.int32)
    if tbm_components.ndim == 1:
        return tbm_components.astype(np.int32)
    if tbm_components.ndim == 2:
        return tbm_components.argmax(axis=1).astype(np.int32)
    raise ValueError(f"Unexpected tbm_components shape: {tbm_components.shape}")


def _downsample_rock(rock_coords: np.ndarray, max_points: int = MAX_ROCK_POINTS) -> np.ndarray:
    if len(rock_coords) <= max_points:
        return rock_coords
    idx = np.linspace(0, len(rock_coords) - 1, max_points, dtype=int)
    return rock_coords[idx]


def _downsample_pair(values: np.ndarray, coords: np.ndarray, max_points: int = MAX_ROCK_POINTS) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) <= max_points:
        return values, coords
    idx = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    return values[idx], coords[idx]


def _build_edge_lines_poly(
    rock_coords: np.ndarray,
    tbm_coords: np.ndarray,
    edge_index: np.ndarray,
    max_edges: int = MAX_EDGE_LINES,
) -> pv.PolyData:
    edge_count = edge_index.shape[1]
    sample_n = min(max_edges, edge_count)
    edge_idx = np.linspace(0, edge_count - 1, sample_n, dtype=int)

    pts = np.zeros((2 * sample_n, 3), dtype=np.float32)
    for i, ei in enumerate(edge_idx):
        ri, ti = edge_index[:, ei]
        pts[2 * i] = rock_coords[ri]
        pts[2 * i + 1] = tbm_coords[ti]

    return pv.lines_from_points(pts)


def _sample_edge_index(edge_index: np.ndarray, max_edges: int = MAX_EDGE_LINES) -> np.ndarray:
    edge_count = edge_index.shape[1]
    if edge_count <= max_edges:
        return edge_index
    idx = np.linspace(0, edge_count - 1, max_edges, dtype=int)
    return edge_index[:, idx]


def _render_3d_panel(
    rock_coords: np.ndarray,
    rock_scalar: np.ndarray,
    tbm_coords: np.ndarray,
    tbm_comp_labels: np.ndarray,
    edge_index_cutter: np.ndarray,
    edge_index_shield: np.ndarray,
    window_size: tuple = PV_WINDOW_SIZE,
) -> np.ndarray:
    """Render a standalone PyVista 3D overview (Panel A)."""
    rock_scalar_plot, rock_plot = _downsample_pair(rock_scalar, rock_coords)

    edge_lines_cutter = _build_edge_lines_poly(
        rock_coords, tbm_coords, edge_index_cutter
    ) if edge_index_cutter.shape[1] > 0 else pv.PolyData()
    edge_lines_shield = _build_edge_lines_poly(
        rock_coords, tbm_coords, edge_index_shield
    ) if edge_index_shield.shape[1] > 0 else pv.PolyData()

    plotter = pv.Plotter(window_size=window_size, off_screen=True)
    rock_poly = pv.PolyData(rock_plot)
    rock_poly["rock_scalar"] = rock_scalar_plot

    plotter.add_mesh(
        rock_poly, scalars="rock_scalar", cmap="turbo", point_size=3.2, opacity=0.28,
        render_points_as_spheres=True, label="Rock voxels",
        scalar_bar_args={"title": r"$V_p$", "fmt": "%.2f", "vertical": True},
    )

    for c in sorted(COMPONENT_COLORS):
        mask = tbm_comp_labels == c
        if not mask.any():
            continue
        plotter.add_mesh(
            tbm_coords[mask], color=COMPONENT_COLORS[c], point_size=7.0, opacity=0.85,
            render_points_as_spheres=True, label=COMPONENT_NAMES[c],
        )

    if edge_lines_cutter.n_points > 0:
        plotter.add_mesh(
            edge_lines_cutter, color="#FF7A1A", line_width=1.0, opacity=0.22,
        )
    if edge_lines_shield.n_points > 0:
        plotter.add_mesh(
            edge_lines_shield, color="#4169E1", line_width=1.0, opacity=0.18,
        )

    plotter.add_legend(bcolor="white", size=(0.18, 0.22), face=None)
    plotter.enable_eye_dome_lighting()

    all_pts = np.vstack([rock_plot, tbm_coords])
    center = all_pts.mean(axis=0)
    diagonal = float(np.linalg.norm(all_pts.ptp(axis=0)))
    pos = center + np.array([diagonal * 0.65, -diagonal * 0.55, diagonal * 0.40])
    plotter.camera.position = pos
    plotter.camera.focal_point = center
    plotter.camera.up = (0.0, 0.0, 1.0)

    plotter.show_bounds(
        grid="back", location="outer",
        xtitle="Chainage (m)", ytitle="Y (m)", ztitle="Z (m)",
        color="#4D4D4D", font_size=10,
    )
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img[..., :3]


def _set_equal_3d(ax, coords: np.ndarray) -> None:
    if coords.size == 0:
        return
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _draw_edges_2d(
    ax,
    rock_coords: np.ndarray,
    tbm_coords: np.ndarray,
    edge_index: np.ndarray,
    coord_idx: tuple[int, int],
    color: str,
    alpha: float = 0.25,
    linewidth: float = 0.5,
) -> None:
    if edge_index.shape[1] == 0:
        return
    sampled = _sample_edge_index(edge_index)
    src = sampled[0]
    dst = sampled[1]
    segments = np.stack(
        [
            rock_coords[src][:, coord_idx],
            tbm_coords[dst][:, coord_idx],
        ],
        axis=1,
    )
    lc = LineCollection(segments, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)


def _hide_ticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _add_component_box(ax, title: str) -> None:
    ax.set_title(title, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CFCFCF")
        spine.set_linewidth(0.8)


def _format_stats_table(ax, cell_text: list[list[str]], col_labels: list[str], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", pad=6, x=0.06)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.62, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.28)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D6D6D6")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#F3F3F3")
            cell.set_text_props(weight="bold")


def plot_publication_graph_figure(
    snapshot,
    rock_scalar: Optional[np.ndarray] = None,
    history_window: Optional[int] = None,
    predict_horizon: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Build a multi-panel publication figure for a single graph snapshot."""
    apply_publication_style()

    data = snapshot.hetero_data
    rock_coords = data["rock"].x[:, :3].cpu().numpy()
    tbm_coords = data["tbm"].x[:, :3].cpu().numpy()
    tbm_comp_labels = snapshot.tbm_components.argmax(dim=1).cpu().numpy()

    if rock_scalar is None:
        rock_scalar = snapshot.rock_attrs[:, 0].cpu().numpy()
        rock_scalar_label = r"Standardized $V_p$"
    else:
        rock_scalar = np.asarray(rock_scalar, dtype=np.float32)
        rock_scalar_label = r"$V_p$"

    if ("rock", "interact", "tbm") in data.edge_types:
        rm_edge_index = data["rock", "interact", "tbm"].edge_index.cpu().numpy()
        edge_attrs = data["rock", "interact", "tbm"]["edge_attrs"]
        distances = np.asarray(edge_attrs["distance"], dtype=np.float32)
        kappas = np.asarray(edge_attrs["kappa"], dtype=np.float32)
        priors = np.asarray(edge_attrs["geometry_prior"], dtype=np.float32)
    else:
        rm_edge_index = np.zeros((2, 0), dtype=np.int64)
        distances = np.zeros(0, dtype=np.float32)
        kappas = np.zeros(0, dtype=np.float32)
        priors = np.zeros(0, dtype=np.float32)

    target_components = tbm_comp_labels[rm_edge_index[1]] if rm_edge_index.shape[1] else np.zeros(0, dtype=np.int32)
    cutter_mask = target_components == 0
    shield_mask = target_components != 0

    rc_edge_index = rm_edge_index[:, cutter_mask] if rm_edge_index.shape[1] else rm_edge_index
    rs_edge_index = rm_edge_index[:, shield_mask] if rm_edge_index.shape[1] else rm_edge_index
    rc_dist = distances[cutter_mask]
    rs_dist = distances[shield_mask]
    rc_kappa = kappas[cutter_mask]
    rs_kappa = kappas[shield_mask]
    rc_prior = priors[cutter_mask]
    rs_prior = priors[shield_mask]

    cutter_nodes = tbm_comp_labels == 0
    shield_nodes = tbm_comp_labels != 0
    rock_idx_cutter = np.unique(rc_edge_index[0]) if rc_edge_index.shape[1] else np.zeros(0, dtype=np.int64)
    rock_idx_shield = np.unique(rs_edge_index[0]) if rs_edge_index.shape[1] else np.zeros(0, dtype=np.int64)

    pv_img = _render_3d_panel(
        rock_coords,
        rock_scalar,
        tbm_coords,
        tbm_comp_labels,
        rc_edge_index,
        rs_edge_index,
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.9, 1.0],
        height_ratios=[1.35, 0.65],
        wspace=0.12,
        hspace=0.14,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(pv_img, aspect="auto")
    ax_a.axis("off")
    ax_a.set_title("3D view of the rock-machine interaction graph at step $t$", loc="left", pad=8, x=0.04)
    add_panel_label(ax_a, "a")
    step_text = [fr"$x_{{face}}(t)$ = {snapshot.chainage:.1f} m"]
    if history_window is not None:
        step_text.append(fr"History window $K$ = {history_window}")
    if predict_horizon is not None:
        step_text.append(fr"Prediction horizon $h$ = {predict_horizon}")
    ax_a.text(
        0.985, 0.985, "\n".join(step_text),
        transform=ax_a.transAxes, ha="right", va="top",
        bbox={"facecolor": "white", "edgecolor": "#D2D2D2", "boxstyle": "round,pad=0.35"},
    )
    ax_a.annotate(
        "Advance direction (+X)",
        xy=(0.53, 0.05), xytext=(0.72, 0.05),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={"arrowstyle": "<|-", "lw": 1.2, "color": "#1A1A1A"},
        ha="center", va="center",
    )

    right_top = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.20)
    cutter_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=right_top[0], wspace=0.08)
    shield_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=right_top[1], wspace=0.08)

    rock_vmin = float(np.percentile(rock_scalar, 5)) if len(rock_scalar) else 0.0
    rock_vmax = float(np.percentile(rock_scalar, 95)) if len(rock_scalar) else 1.0
    norm = mcolors.Normalize(vmin=rock_vmin, vmax=rock_vmax)
    cmap = plt.get_cmap("turbo")

    ax_b1 = fig.add_subplot(cutter_grid[0, 0])
    ax_b1.scatter(
        rock_coords[rock_idx_cutter, 1], rock_coords[rock_idx_cutter, 2],
        c=rock_scalar[rock_idx_cutter], cmap=cmap, norm=norm, s=6, alpha=0.65, linewidths=0,
    )
    ax_b1.scatter(
        tbm_coords[cutter_nodes, 1], tbm_coords[cutter_nodes, 2],
        s=20, c="#FF7A1A", edgecolors="white", linewidths=0.3,
    )
    _draw_edges_2d(ax_b1, rock_coords, tbm_coords, rc_edge_index, (1, 2), "#FF7A1A", alpha=0.18)
    ax_b1.set_aspect("equal")
    _hide_ticks(ax_b1)
    _add_component_box(ax_b1, "Front view")

    ax_b2 = fig.add_subplot(cutter_grid[0, 1])
    ax_b2.scatter(
        rock_coords[rock_idx_cutter, 0], rock_coords[rock_idx_cutter, 2],
        c=rock_scalar[rock_idx_cutter], cmap=cmap, norm=norm, s=6, alpha=0.65, linewidths=0,
    )
    ax_b2.scatter(
        tbm_coords[cutter_nodes, 0], tbm_coords[cutter_nodes, 2],
        s=20, c="#FF7A1A", edgecolors="white", linewidths=0.3,
    )
    _draw_edges_2d(ax_b2, rock_coords, tbm_coords, rc_edge_index, (0, 2), "#FF7A1A", alpha=0.18)
    ax_b2.set_aspect("equal")
    _hide_ticks(ax_b2)
    _add_component_box(ax_b2, "Side slice ($X$-$Z$)")
    add_panel_label(ax_b1, "b")
    ax_b1.text(0.00, 1.08, "Local view of cutterhead interaction", transform=ax_b1.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")

    ax_c1 = fig.add_subplot(shield_grid[0, 0])
    ax_c1.scatter(
        rock_coords[rock_idx_shield, 1], rock_coords[rock_idx_shield, 2],
        c=rock_scalar[rock_idx_shield], cmap=cmap, norm=norm, s=6, alpha=0.45, linewidths=0,
    )
    ax_c1.scatter(
        tbm_coords[shield_nodes, 1], tbm_coords[shield_nodes, 2],
        s=12, c="#4169E1", edgecolors="white", linewidths=0.25,
    )
    _draw_edges_2d(ax_c1, rock_coords, tbm_coords, rs_edge_index, (1, 2), "#4169E1", alpha=0.14)
    ax_c1.set_aspect("equal")
    _hide_ticks(ax_c1)
    _add_component_box(ax_c1, "Cross-section ($Y$-$Z$)")

    ax_c2 = fig.add_subplot(shield_grid[0, 1])
    ax_c2.scatter(
        rock_coords[rock_idx_shield, 0], rock_coords[rock_idx_shield, 2],
        c=rock_scalar[rock_idx_shield], cmap=cmap, norm=norm, s=6, alpha=0.45, linewidths=0,
    )
    ax_c2.scatter(
        tbm_coords[shield_nodes, 0], tbm_coords[shield_nodes, 2],
        s=12, c="#4169E1", edgecolors="white", linewidths=0.25,
    )
    _draw_edges_2d(ax_c2, rock_coords, tbm_coords, rs_edge_index, (0, 2), "#4169E1", alpha=0.14)
    ax_c2.set_aspect("equal")
    _hide_ticks(ax_c2)
    _add_component_box(ax_c2, "Side view ($X$-$Z$)")
    add_panel_label(ax_c1, "c")
    ax_c1.text(0.00, 1.08, "Local view of shield interaction", transform=ax_c1.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")

    cax = fig.add_axes([0.957, 0.66, 0.010, 0.17])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(rock_scalar_label)

    bottom = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[1, :], wspace=0.18)

    ax_d0 = fig.add_subplot(bottom[0, 0])
    node_table = [
        ["Active rock voxels", f"{len(rock_coords):,}"],
        ["Rock voxels near cutterhead", f"{len(rock_idx_cutter):,}"],
        ["Rock voxels near shield", f"{len(rock_idx_shield):,}"],
        ["Cutterhead surface nodes", f"{int(cutter_nodes.sum()):,}"],
        ["Shield surface nodes", f"{int(shield_nodes.sum()):,}"],
        ["Total nodes", f"{len(rock_coords) + len(tbm_coords):,}"],
    ]
    _format_stats_table(ax_d0, node_table, ["Node statistics", "Count"], "Statistics of the constructed graph at step $t$")
    add_panel_label(ax_d0, "d")

    ax_d1 = fig.add_subplot(bottom[0, 1])
    edge_table = [
        [r"Rock-rock edges ($E_{rr}$)", f"{data['rock', 'adj', 'rock'].edge_index.shape[1]:,}"],
        [r"Shield-structure edges ($E_{ss}$)", f"{data['tbm', 'adj', 'tbm'].edge_index.shape[1]:,}"],
        [r"Rock-cutterhead edges ($E_{rc}$)", f"{rc_edge_index.shape[1]:,}"],
        [r"Rock-shield edges ($E_{rs}$)", f"{rs_edge_index.shape[1]:,}"],
        [r"Total rock-machine edges ($E_{rm}$)", f"{rm_edge_index.shape[1]:,}"],
    ]
    _format_stats_table(ax_d1, edge_table, ["Edge statistics", "Count"], "")

    ax_d2 = fig.add_subplot(bottom[0, 2])
    ax_d2.set_title("Rock-machine edges by target")
    sizes = np.array([rc_edge_index.shape[1], rs_edge_index.shape[1]], dtype=float)
    if sizes.sum() > 0:
        wedges, _ = ax_d2.pie(
            sizes,
            startangle=90,
            colors=["#FF7A1A", "#4169E1"],
            wedgeprops={"width": 0.45, "edgecolor": "white"},
        )
        shares = sizes / sizes.sum()
        ax_d2.legend(
            wedges,
            [fr"To cutterhead ({shares[0] * 100:.1f}%)", fr"To shield ({shares[1] * 100:.1f}%)"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.10),
            fontsize=8,
        )
    ax_d2.set_aspect("equal")

    hist_specs = [
        ("Edge length $d_{ij}$ (m)", rc_dist, rs_dist, (0, max(np.max(distances, initial=1.0), 1.0))),
        ("Normal consistency $\\kappa_{ij}$", rc_kappa, rs_kappa, (0, 1)),
        ("Geometric prior $\\pi^{rm}_{ij}$", rc_prior, rs_prior, (0, 1)),
    ]
    for i, (title, a_vals, b_vals, xlim) in enumerate(hist_specs, start=3):
        ax = fig.add_subplot(bottom[0, i])
        bins = np.linspace(xlim[0], xlim[1], 28)
        if len(a_vals):
            ax.hist(a_vals, bins=bins, density=True, color="#FF7A1A", alpha=0.50, edgecolor="#FF7A1A", linewidth=0.4, label=r"$E_{rc}$")
        if len(b_vals):
            ax.hist(b_vals, bins=bins, density=True, color="#4169E1", alpha=0.38, edgecolor="#4169E1", linewidth=0.4, label=r"$E_{rs}$")
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylabel("Density")
        ax.grid(True, axis="y")
        if i == 5:
            ax.legend(loc="upper right", fontsize=8)

    fig.subplots_adjust(left=0.03, right=0.95, top=0.965, bottom=0.05)
    save_figure(fig, save_path)


# ── IJGIS-Style 4-Panel Graph Construction Figure ──

def _draw_panel_a_schematic(
    ax: plt.Axes,
    cutterhead_radius: float = 4.0,
    shield_radius: float = 3.95,
    front_len: float = 3.0,
    middle_len: float = 3.5,
    tail_len: float = 3.5,
    cutterhead_look_ahead: float = 5.0,
    tau: float = 2.0,
) -> None:
    """Panel a: 2D longitudinal profile schematic showing spatial entities.

    Draws a side-view (X-Z plane) schematic with:
    - TBM outline (cutterhead disk + shield cylinder)
    - Active zone annotations (cutterhead look-ahead + shield annular)
    - Excavated zone shading
    - Rock voxel grid indication
    """
    total_shield = front_len + middle_len + tail_len
    x_tail = -total_shield
    x_face = 0.0

    # Excavated zone (gray shading behind TBM)
    ax.axvspan(x_tail - 2, x_face, color="#E8E8E8", zorder=0)
    ax.text((x_tail + x_face) / 2, -cutterhead_radius - 1.8, "Excavated",
            ha="center", va="top", fontsize=6, color="#888888", style="italic")

    # Rock voxel grid indication (ahead of face)
    grid_x = np.linspace(x_face + 0.5, x_face + cutterhead_look_ahead + 2, 8)
    grid_z = np.linspace(-cutterhead_radius - 1, cutterhead_radius + 1, 7)
    for gx in grid_x:
        ax.plot([gx, gx], [grid_z[0], grid_z[-1]], color="#C0C0C0",
                linewidth=0.3, zorder=1)
    for gz in grid_z:
        ax.plot([grid_x[0], grid_x[-1]], [gz, gz], color="#C0C0C0",
                linewidth=0.3, zorder=1)
    # Scatter some "active" voxels in the cutterhead zone
    np.random.seed(42)
    n_demo = 40
    demo_x = np.random.uniform(x_face, x_face + cutterhead_look_ahead, n_demo)
    demo_z = np.random.uniform(-cutterhead_radius, cutterhead_radius, n_demo)
    ax.scatter(demo_x, demo_z, s=4, c=PALETTE["rock"], alpha=0.5, zorder=2)

    # TBM outline
    # Cutterhead (vertical line at x=0)
    ax.plot([x_face, x_face], [-cutterhead_radius, cutterhead_radius],
            color=COMPONENT_COLORS[0], linewidth=2.5, zorder=5)
    # Shield top and bottom lines
    ax.plot([x_tail, x_face], [shield_radius, shield_radius],
            color=COMPONENT_COLORS[1], linewidth=1.5, zorder=5)
    ax.plot([x_tail, x_face], [-shield_radius, -shield_radius],
            color=COMPONENT_COLORS[1], linewidth=1.5, zorder=5)
    # Component boundaries (vertical dashed)
    for bx, label in [(x_tail, "Tail"), (-front_len - middle_len, "Mid"),
                       (-front_len, "Front")]:
        ax.plot([bx, bx], [-shield_radius, shield_radius],
                color="#AAAAAA", linewidth=0.6, linestyle="--", zorder=4)
        ax.text(bx, shield_radius + 0.3, label, ha="center", va="bottom",
                fontsize=5.5, color="#666666")
    ax.text(x_face, cutterhead_radius + 0.3, "Cutterhead", ha="center",
            va="bottom", fontsize=5.5, color=COMPONENT_COLORS[0], fontweight="bold")

    # Active zone: cutterhead look-ahead
    ax.axvspan(x_face, x_face + cutterhead_look_ahead,
               color="#FF7A1A", alpha=0.10, zorder=1)
    ax.annotate(
        "", xy=(x_face + cutterhead_look_ahead, cutterhead_radius + 0.8),
        xytext=(x_face, cutterhead_radius + 0.8),
        arrowprops={"arrowstyle": "<->", "color": "#FF7A1A", "lw": 0.8},
    )
    ax.text(x_face + cutterhead_look_ahead / 2, cutterhead_radius + 1.1,
            r"$L_f$", ha="center", va="bottom", fontsize=7, color="#FF7A1A")

    # Active zone: shield annular
    ax.fill_between(
        [x_tail, x_face],
        shield_radius, shield_radius + tau,
        color="#4169E1", alpha=0.12, zorder=1,
    )
    ax.fill_between(
        [x_tail, x_face],
        -shield_radius, -shield_radius - tau,
        color="#4169E1", alpha=0.12, zorder=1,
    )
    # τ annotation
    mid_x = (x_tail + x_face) / 2
    ax.annotate(
        "", xy=(mid_x, shield_radius + tau),
        xytext=(mid_x, shield_radius),
        arrowprops={"arrowstyle": "<->", "color": "#4169E1", "lw": 0.8},
    )
    ax.text(mid_x + 0.4, shield_radius + tau / 2, r"$\tau$",
            fontsize=7, color="#4169E1", va="center")

    # Advance direction arrow
    ax.annotate(
        "Advance", xy=(x_face + cutterhead_look_ahead + 2.5, 0),
        xytext=(x_face + cutterhead_look_ahead + 0.5, 0),
        arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 1.0},
        fontsize=6, va="center", color="#333333",
    )

    # x_face label
    ax.annotate(
        r"$x_{face}$", xy=(x_face, -cutterhead_radius - 1.2),
        fontsize=7, ha="center", va="top", color="#333333",
    )

    ax.set_xlim(x_tail - 3, x_face + cutterhead_look_ahead + 3)
    ax.set_ylim(-cutterhead_radius - 2.5, cutterhead_radius + 2.5)
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")


def _draw_panel_b_crosssection(
    ax: plt.Axes,
    shield_radius: float = 3.95,
    tau: float = 2.0,
) -> None:
    """Panel b: 2D cross-section schematic showing edge construction constraints.

    Draws a Y-Z cross-section showing:
    - Shield circle
    - Annular active zone
    - Candidate edges with geometry prior filtering
    - Normal consistency κ illustration
    """
    theta = np.linspace(0, 2 * np.pi, 200)

    # Shield circle
    ax.plot(shield_radius * np.cos(theta), shield_radius * np.sin(theta),
            color=COMPONENT_COLORS[1], linewidth=1.5, zorder=5)
    ax.text(0, 0, "TBM\nshield", ha="center", va="center",
            fontsize=6, color="#888888")

    # Annular active zone
    outer_r = shield_radius + tau
    ax.fill_between(
        outer_r * np.cos(theta), outer_r * np.sin(theta),
        shield_radius * np.cos(theta), shield_radius * np.sin(theta),
        color="#4169E1", alpha=0.10, zorder=1,
    )
    ax.plot(outer_r * np.cos(theta), outer_r * np.sin(theta),
            color="#4169E1", linewidth=0.8, linestyle="--", zorder=3)

    # τ annotation
    ax.annotate(
        "", xy=(outer_r, 0), xytext=(shield_radius, 0),
        arrowprops={"arrowstyle": "<->", "color": "#4169E1", "lw": 0.8},
    )
    ax.text((shield_radius + outer_r) / 2, -0.5, r"$\tau$",
            fontsize=7, color="#4169E1", ha="center", va="top")

    # Demo rock voxels in annular zone
    np.random.seed(7)
    n_rock = 12
    rock_angles = np.random.uniform(0, 2 * np.pi, n_rock)
    rock_radii = np.random.uniform(shield_radius + 0.3, outer_r - 0.2, n_rock)
    rock_y = rock_radii * np.cos(rock_angles)
    rock_z = rock_radii * np.sin(rock_angles)
    ax.scatter(rock_y, rock_z, s=12, c=PALETTE["rock"], alpha=0.7,
               zorder=4, edgecolors="white", linewidths=0.3)

    # Draw a few candidate edges with geometry prior illustration
    # Pick 3 edges: one with high κ, one with low κ, one filtered out
    demo_pairs = [
        (0, 0, "#4169E1", 1.0, r"$\kappa \approx 1$"),    # normal to surface
        (3, 3, "#FF7A1A", 0.5, r"$\kappa \approx 0.5$"),   # oblique
        (6, 6, "#CCCCCC", 0.3, r"Filtered ($\pi^{rm} < \eta_{min}$)"),  # filtered
    ]
    for ri, ti, color, alpha, label in demo_pairs:
        ry, rz = rock_y[ri], rock_z[ri]
        # TBM surface point closest to this rock voxel
        tbm_y = shield_radius * np.cos(rock_angles[ri])
        tbm_z = shield_radius * np.sin(rock_angles[ri])
        ax.plot([ry, tbm_y], [rz, tbm_z], color=color, linewidth=0.8,
                alpha=alpha, zorder=3)

    # Normal vector illustration on one TBM node
    idx_norm = 0
    tbm_y_n = shield_radius * np.cos(rock_angles[idx_norm])
    tbm_z_n = shield_radius * np.sin(rock_angles[idx_norm])
    norm_y = np.cos(rock_angles[idx_norm])
    norm_z = np.sin(rock_angles[idx_norm])
    ax.annotate(
        "", xy=(tbm_y_n + 1.2 * norm_y, tbm_z_n + 1.2 * norm_z),
        xytext=(tbm_y_n, tbm_z_n),
        arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 0.7},
    )
    ax.text(tbm_y_n + 1.5 * norm_y, tbm_z_n + 1.5 * norm_z, r"$\mathbf{n}_j$",
            fontsize=6, ha="center", va="center", color="#333333")

    # Legend for edge types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#4169E1", linewidth=1.0, label=r"High $\kappa$ (retained)"),
        Line2D([0], [0], color="#FF7A1A", linewidth=1.0, label=r"Medium $\kappa$ (retained)"),
        Line2D([0], [0], color="#CCCCCC", linewidth=1.0, linestyle=":",
               label=r"Low $\pi^{rm}$ (filtered)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=5.5,
              framealpha=0.9, edgecolor="#CCCCCC")

    # R_s label
    ax.annotate(
        r"$R_s$", xy=(shield_radius / 2, 0.4),
        fontsize=7, color=COMPONENT_COLORS[1], ha="center",
    )

    ax.set_xlim(-outer_r - 1.5, outer_r + 1.5)
    ax.set_ylim(-outer_r - 1.5, outer_r + 1.5)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")


def _draw_panel_c_adjacency(
    ax: plt.Axes,
    snapshot,
    max_nodes: int = 200,
) -> None:
    """Panel c: Adjacency matrix heatmap showing graph snapshot structure.

    Displays the block structure: E_rr (top-left), E_mm (bottom-right),
    E_rm (off-diagonal blocks).
    """
    data = snapshot.hetero_data

    # Get edge indices
    rr_edges = data["rock", "adj", "rock"].edge_index.cpu().numpy()
    mm_edges = data["tbm", "adj", "tbm"].edge_index.cpu().numpy()
    n_rock = data["rock"].x.shape[0]
    n_tbm = data["tbm"].x.shape[0]

    # Get rock-machine edges
    if ("rock", "interact", "tbm") in data.edge_types:
        rm_edges = data["rock", "interact", "tbm"].edge_index.cpu().numpy()
        # Get geometry prior for edge weights
        edge_attrs = data["rock", "interact", "tbm"]["edge_attrs"]
        priors = np.asarray(edge_attrs["geometry_prior"], dtype=np.float32)
    else:
        rm_edges = np.zeros((2, 0), dtype=np.int64)
        priors = np.zeros(0, dtype=np.float32)

    # Subsample if too large
    if n_rock > max_nodes:
        rock_idx = np.linspace(0, n_rock - 1, max_nodes, dtype=int)
        rr_mask = np.isin(rr_edges[0], rock_idx) & np.isin(rr_edges[1], rock_idx)
        rr_edges = rr_edges[:, rr_mask]
        # Remap indices
        rock_map = {old: new for new, old in enumerate(rock_idx)}
        rr_edges = np.array([[rock_map.get(s, s) for s in rr_edges[0]],
                             [rock_map.get(d, d) for d in rr_edges[1]]])
        n_rock_sub = max_nodes
    else:
        n_rock_sub = n_rock
        rock_idx = np.arange(n_rock)

    if n_tbm > max_nodes:
        tbm_idx = np.linspace(0, n_tbm - 1, max_nodes, dtype=int)
        mm_mask = np.isin(mm_edges[0], tbm_idx) & np.isin(mm_edges[1], tbm_idx)
        mm_edges = mm_edges[:, mm_mask]
        tbm_map = {old: new for new, old in enumerate(tbm_idx)}
        mm_edges = np.array([[tbm_map.get(s, s) for s in mm_edges[0]],
                             [tbm_map.get(d, d) for d in mm_edges[1]]])
        n_tbm_sub = max_nodes
    else:
        n_tbm_sub = n_tbm
        tbm_idx = np.arange(n_tbm)

    # Build full adjacency matrix
    N = n_rock_sub + n_tbm_sub
    adj = np.zeros((N, N), dtype=np.float32)

    # E_rr block (binary)
    valid_rr = (rr_edges[0] < n_rock_sub) & (rr_edges[1] < n_rock_sub)
    adj[rr_edges[0][valid_rr], rr_edges[1][valid_rr]] = 0.3

    # E_mm block (binary)
    valid_mm = (mm_edges[0] < n_tbm_sub) & (mm_edges[1] < n_tbm_sub)
    adj[n_rock_sub + mm_edges[0][valid_mm],
        n_rock_sub + mm_edges[1][valid_mm]] = 0.3

    # E_rm block (weighted by geometry prior)
    if rm_edges.shape[1] > 0:
        # Filter to subsampled indices
        rm_rock_valid = np.isin(rm_edges[0], rock_idx) if n_rock > max_nodes else np.ones(rm_edges.shape[1], dtype=bool)
        rm_tbm_valid = np.isin(rm_edges[1], tbm_idx) if n_tbm > max_nodes else np.ones(rm_edges.shape[1], dtype=bool)
        rm_valid = rm_rock_valid & rm_tbm_valid
        rm_src = rm_edges[0][rm_valid]
        rm_dst = rm_edges[1][rm_valid]
        rm_prior = priors[rm_valid]

        # Remap
        if n_rock > max_nodes:
            rm_src = np.array([rock_map.get(s, s) for s in rm_src])
        if n_tbm > max_nodes:
            rm_dst = np.array([tbm_map.get(d, d) for d in rm_dst])

        valid_rm = (rm_src < n_rock_sub) & (rm_dst < n_tbm_sub)
        adj[rm_src[valid_rm], n_rock_sub + rm_dst[valid_rm]] = np.clip(rm_prior[valid_rm], 0.3, 1.0)
        adj[n_rock_sub + rm_dst[valid_rm], rm_src[valid_rm]] = np.clip(rm_prior[valid_rm], 0.3, 1.0)

    # Plot
    from matplotlib.colors import ListedColormap
    # Custom colormap: white → light blue → dark blue
    cmap_adj = ListedColormap(["#FFFFFF", "#D4E6F1", "#85C1E9", "#2E86C1", "#1B4F72"])

    im = ax.imshow(adj, cmap=cmap_adj, vmin=0, vmax=1.0,
                   interpolation="nearest", aspect="auto")

    # Block boundary lines
    ax.axhline(n_rock_sub - 0.5, color="#333333", linewidth=0.8)
    ax.axvline(n_rock_sub - 0.5, color="#333333", linewidth=0.8)

    # Block labels
    ax.text(n_rock_sub / 2, -3, r"$E_{rr}$", ha="center", va="bottom",
            fontsize=7, color=PALETTE["rock"])
    ax.text(n_rock_sub + n_tbm_sub / 2, -3, r"$E_{mm}$", ha="center", va="bottom",
            fontsize=7, color=PALETTE["tbm"])
    ax.text(-3, n_rock_sub / 2, r"$V^r$", ha="right", va="center",
            fontsize=7, color=PALETTE["rock"], rotation=90)
    ax.text(-3, n_rock_sub + n_tbm_sub / 2, r"$V^m$", ha="right", va="center",
            fontsize=7, color=PALETTE["tbm"], rotation=90)

    # E_rm label in off-diagonal
    ax.text(n_rock_sub + n_tbm_sub * 0.3, n_rock_sub * 0.3,
            r"$E_{rm}$", ha="center", va="center",
            fontsize=6, color="#2E86C1", alpha=0.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1})

    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")
    # Reduce tick density
    ax.set_xticks(np.linspace(0, N - 1, 5, dtype=int))
    ax.set_yticks(np.linspace(0, N - 1, 5, dtype=int))

    return im


def _draw_panel_d_evolution(
    ax: plt.Axes,
    snapshots: list,
    chainage_indices: list[int],
    max_nodes: int = 100,
) -> None:
    """Panel d: Graph sequence evolution — 3 representative chainages side-by-side.

    Shows how the adjacency structure changes across excavation steps.
    """
    n_steps = len(chainage_indices)
    if n_steps == 0:
        return

    sub_axes = []
    for i, step_idx in enumerate(chainage_indices):
        if step_idx >= len(snapshots):
            continue
        snap = snapshots[step_idx]
        data = snap.hetero_data

        n_rock = data["rock"].x.shape[0]
        n_tbm = data["tbm"].x.shape[0]

        # Build small adjacency for visualization
        rr_edges = data["rock", "adj", "rock"].edge_index.cpu().numpy()
        mm_edges = data["tbm", "adj", "tbm"].edge_index.cpu().numpy()
        if ("rock", "interact", "tbm") in data.edge_types:
            rm_edges = data["rock", "interact", "tbm"].edge_index.cpu().numpy()
        else:
            rm_edges = np.zeros((2, 0), dtype=np.int64)

        # Subsample
        nr = min(n_rock, max_nodes)
        nt = min(n_tbm, max_nodes)
        N = nr + nt
        adj = np.zeros((N, N), dtype=np.float32)

        # Simplified: just mark existence
        rr_sub = rr_edges[:, (rr_edges[0] < nr) & (rr_edges[1] < nr)]
        if rr_sub.shape[1] > 0:
            adj[rr_sub[0], rr_sub[1]] = 0.3

        mm_sub = mm_edges[:, (mm_edges[0] < nt) & (mm_edges[1] < nt)]
        if mm_sub.shape[1] > 0:
            adj[nr + mm_sub[0], nr + mm_sub[1]] = 0.3

        rm_sub_src = rm_edges[0][rm_edges[0] < nr]
        rm_sub_dst = rm_edges[1][rm_edges[1] < nt]
        if len(rm_sub_src) > 0:
            adj[rm_sub_src, nr + rm_sub_dst] = 0.7
            adj[nr + rm_sub_dst, rm_sub_src] = 0.7

        # Create inset axes
        left = i / n_steps + 0.02
        width = 1.0 / n_steps - 0.04
        inset_ax = ax.inset_axes([left, 0.05, width, 0.85])

        from matplotlib.colors import ListedColormap
        cmap_adj = ListedColormap(["#FFFFFF", "#D4E6F1", "#85C1E9", "#2E86C1", "#1B4F72"])
        inset_ax.imshow(adj, cmap=cmap_adj, vmin=0, vmax=1.0,
                        interpolation="nearest", aspect="auto")
        inset_ax.axhline(nr - 0.5, color="#333333", linewidth=0.5)
        inset_ax.axvline(nr - 0.5, color="#333333", linewidth=0.5)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title(f"Ch. {snap.chainage:.0f} m", fontsize=6, pad=2)

        # Edge counts
        n_rr = rr_edges.shape[1]
        n_mm = mm_edges.shape[1]
        n_rm = rm_edges.shape[1]
        inset_ax.text(0.5, -0.08,
                      f"|E|={n_rr + n_mm + n_rm:,}",
                      transform=inset_ax.transAxes, ha="center",
                      fontsize=5, color="#666666")

        sub_axes.append(inset_ax)

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_graph_construction_figure(
    snapshot,
    snapshots: Optional[list] = None,
    chainage_indices: Optional[list[int]] = None,
    cutterhead_radius: float = 4.0,
    shield_radius: float = 3.95,
    front_len: float = 3.0,
    middle_len: float = 3.5,
    tail_len: float = 3.5,
    cutterhead_look_ahead: float = 5.0,
    tau: float = 2.0,
    save_path: Optional[str] = None,
):
    """IJGIS-style 4-panel graph construction figure.

    Panels:
      a — Spatial entity definition (2D longitudinal profile schematic)
      b — Edge construction constraints (2D cross-section schematic)
      c — Graph snapshot structure (adjacency matrix heatmap)
      d — Graph sequence evolution (multi-chainage adjacency comparison)

    Args:
        snapshot: GraphSnapshot for panels a-c.
        snapshots: full list of GraphSnapshots for panel d (optional).
        chainage_indices: indices into snapshots for panel d (default: early/mid/late).
        cutterhead_radius/shield_radius/front_len/middle_len/tail_len: TBM geometry.
        cutterhead_look_ahead: L_f parameter for active zone.
        tau: annular zone thickness for active zone.
        save_path: output file path.
    """
    apply_publication_style()

    fig = plt.figure(figsize=figure_size("double", aspect=0.75))
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.30)

    # Panel a: longitudinal profile
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_panel_a_schematic(
        ax_a, cutterhead_radius, shield_radius, front_len,
        middle_len, tail_len, cutterhead_look_ahead, tau,
    )
    add_panel_label(ax_a, "a")

    # Panel b: cross-section
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_panel_b_crosssection(ax_b, shield_radius, tau)
    add_panel_label(ax_b, "b")

    # Panel c: adjacency matrix
    ax_c = fig.add_subplot(gs[1, 0])
    _draw_panel_c_adjacency(ax_c, snapshot)
    add_panel_label(ax_c, "c")

    # Panel d: graph sequence evolution
    ax_d = fig.add_subplot(gs[1, 1])
    if snapshots is not None and len(snapshots) > 1:
        if chainage_indices is None:
            n = len(snapshots)
            chainage_indices = [0, n // 2, n - 1]
        _draw_panel_d_evolution(ax_d, snapshots, chainage_indices)
    else:
        ax_d.text(0.5, 0.5, "Graph sequence\nnot provided",
                  transform=ax_d.transAxes, ha="center", va="center",
                  fontsize=8, color="#999999")
        ax_d.axis("off")
    add_panel_label(ax_d, "d")

    fig.tight_layout()
    save_figure(fig, save_path)


# ── Public API ──

def plot_graph_snapshot(rock_coords: np.ndarray,
                        tbm_positions: np.ndarray,
                        rm_edge_index: np.ndarray,
                        tbm_components: Optional[np.ndarray] = None,
                        x_slice: Optional[float] = None,
                        save_path: Optional[str] = None):
    """Plot a publication-ready graph snapshot figure.

    Panel A (3D overview) via PyVista, Panels B (YZ cross-section) and
    C (edge distance histogram) via matplotlib.
    """
    tbm_comp_labels = _prepare_tbm_components(tbm_components, len(tbm_positions))

    # Phase 1: PyVista 3D panel
    pv_img = _render_3d_panel(rock_coords, tbm_positions, tbm_comp_labels, rm_edge_index)

    # Downsample rock for 2D panels
    rock_plot = _downsample_rock(rock_coords)

    # Phase 2: Matplotlib assembly
    apply_publication_style()
    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])

    # Panel A: PyVista screenshot embedded
    ax_a = fig.add_subplot(gs[:, 0])
    ax_a.imshow(pv_img, aspect="auto")
    ax_a.axis("off")

    # Panel B: YZ cross-section (matplotlib 2D scatter)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(
        rock_plot[:, 1], rock_plot[:, 2],
        s=3, c=PALETTE["rock"], alpha=0.20, linewidths=0,
    )
    for c in sorted(COMPONENT_COLORS):
        mask = tbm_comp_labels == c
        if not mask.any():
            continue
        ax_b.scatter(
            tbm_positions[mask, 1], tbm_positions[mask, 2],
            s=8, c=COMPONENT_COLORS[c], alpha=0.9, linewidths=0,
        )
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("Y (m)")
    ax_b.set_ylabel("Z (m)")
    ax_b.set_title("Cross-Section View")
    ax_b.grid(True)
    add_panel_label(ax_b, "B")

    # Panel C: edge distance histogram
    ax_hist = fig.add_subplot(gs[1, 1])

    if rm_edge_index.shape[1] > 0:
        src = rm_edge_index[0]
        dst = rm_edge_index[1]
        dists = np.linalg.norm(rock_coords[src] - tbm_positions[dst], axis=1)
        bins = np.linspace(dists.min(), dists.max(), 30)
        ax_hist.hist(
            dists, bins=bins, density=True,
            color=PALETTE["full"], alpha=0.75, edgecolor="white", linewidth=0.6,
        )
        ax_cdf = ax_hist.twinx()
        x_sorted = np.sort(dists)
        y_cdf = np.linspace(0, 1, len(x_sorted))
        ax_cdf.plot(x_sorted, y_cdf, color=PALETTE["baseline"], linewidth=1.4)
        ax_cdf.set_ylabel("Cumulative probability")
        ax_cdf.spines["top"].set_visible(False)
        ax_cdf.tick_params(axis="y", labelsize=8)

        stats_text = (
            f"$|E_{{rm}}|$ = {len(dists):,}\n"
            f"Mean = {dists.mean():.2f} m\n"
            f"P90 = {np.percentile(dists, 90):.2f} m"
        )
        ax_hist.text(
            0.98, 0.95, stats_text,
            transform=ax_hist.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.3"},
        )
    ax_hist.set_xlabel("Rock-TBM edge distance (m)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Interaction Distance Distribution")
    ax_hist.grid(True)
    add_panel_label(ax_hist, "C")

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_edge_statistics(snapshots: list, save_path: Optional[str] = None):
    """Plot journal-style edge statistics over chainage."""
    apply_publication_style()
    chainages = np.asarray([s.chainage for s in snapshots], dtype=float)
    n_rm_edges = []
    mean_dists = []

    for snap in snapshots:
        data = snap.hetero_data
        if ("rock", "interact", "tbm") in data.edge_types and "edge_attrs" in data["rock", "interact", "tbm"]:
            dists = np.asarray(data["rock", "interact", "tbm"]["edge_attrs"]["distance"], dtype=float)
            n_rm_edges.append(len(dists))
            mean_dists.append(float(dists.mean()) if len(dists) else 0.0)
        else:
            n_rm_edges.append(0)
            mean_dists.append(0.0)

    n_rm_edges = np.asarray(n_rm_edges, dtype=float)
    mean_dists = np.asarray(mean_dists, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=True)

    axes[0].plot(chainages, n_rm_edges, color=PALETTE["full"], marker="o", markersize=3)
    axes[0].fill_between(chainages, n_rm_edges, color=PALETTE["full"], alpha=0.12)
    axes[0].set_ylabel(r"Number of $E_{rm}$ edges")
    axes[0].set_title("Evolution of Graph Interaction Statistics", pad=8)
    axes[0].grid(True)
    add_panel_label(axes[0], "A")

    axes[1].plot(chainages, mean_dists, color=PALETTE["baseline"], marker="o", markersize=3)
    axes[1].fill_between(chainages, mean_dists, color=PALETTE["baseline"], alpha=0.14)
    axes[1].set_ylabel("Mean distance (m)")
    axes[1].set_xlabel("Chainage (m)")
    axes[1].grid(True)
    add_panel_label(axes[1], "B")

    fig.tight_layout()
    save_figure(fig, save_path)
