from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tbm_shield_sticking"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

EVENT_START = pd.Timestamp("2024-08-03 13:00:00")
EVENT_CONFIRM = pd.Timestamp("2024-08-03 16:30:00")
EVENT_CHAINAGE = 1017205.0
FAULT_START = 1017277.0
FAULT_END = 1017477.0
TBM_RADIUS = 4.0

COMPONENT_COLORS = {
    "cutterhead": "#6b7280",
    "front_shield": "#4daf4a",
    "middle_shield": "#f28e2b",
    "tail_shield": "#4e79a7",
}


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rel_chainage(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return values - EVENT_CHAINAGE


def build_table_1() -> None:
    rows = [
        {
            "Data group": "Event log",
            "Required fields": "confirmed stuck time, event chainage, event window, engineering description",
            "Role in the experiment": "Defines the shield-sticking event and evaluation window",
            "Source files": "event_metadata.csv",
        },
        {
            "Data group": "TSP geological data",
            "Required fields": "voxel coordinates, Vp, Vs, geological anomaly score q_i",
            "Role in the experiment": "Provides rock voxels and geological anomaly scores",
            "Source files": "tsp_voxels.csv",
        },
        {
            "Data group": "TBM monitoring",
            "Required fields": "shield pressure, shield displacement, thrust, advance rate, RPM, timestamp, chainage",
            "Role in the experiment": "Describes machine response during the event",
            "Source files": "tbm_monitoring.csv",
        },
        {
            "Data group": "TBM geometry",
            "Required fields": "component-labelled cutterhead, front shield, middle shield and tail shield surface nodes",
            "Role in the experiment": "Provides component-labelled TBM surface nodes",
            "Source files": "tbm_surface_nodes.csv",
        },
        {
            "Data group": "Alignment data",
            "Required fields": "timestamp-chainage mapping, TSP local coordinate, event chainage",
            "Role in the experiment": "Links geological field, monitoring records and graph snapshots",
            "Source files": "alignment_map.csv",
        },
        {
            "Data group": "Graph provenance",
            "Required fields": "rock voxel id, TBM node id, component label, distance, normal compatibility, edge weight",
            "Role in the experiment": "Traces component-level interpretation back to specific rock-TBM edges",
            "Source files": "graph_edges_event_sample_top500.csv; edge_provenance_top50.csv",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "table1_case_data_alignment_summary.csv", index=False)
    with (TABLE_DIR / "table1_case_data_alignment_summary.md").open("w", encoding="utf-8") as f:
        f.write("Table 1. Case event, input data and alignment used for event-centred evaluation.\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")


def add_tbm_side(ax: plt.Axes, face_chainage: float, y0: float = -4.0, y1: float = 4.0, alpha: float = 0.9) -> None:
    nodes = pd.read_csv(DATA_DIR / "tbm_surface_nodes.csv")
    for comp, g in nodes.groupby("component"):
        x_min = rel_chainage(face_chainage + g["local_x_m"].min())
        x_max = rel_chainage(face_chainage + g["local_x_m"].max())
        width = max(x_max - x_min, 0.15)
        ax.add_patch(
            plt.Rectangle(
                (x_min, y0),
                width,
                y1 - y0,
                facecolor=COMPONENT_COLORS.get(comp, "#999999"),
                edgecolor="#222222",
                linewidth=0.8,
                alpha=alpha,
            )
        )
        ax.text((x_min + x_max) / 2, y1 + 0.35, comp.replace("_", "\n"), ha="center", va="bottom", fontsize=7)


def figure_1_spatial_setting() -> None:
    tsp = pd.read_csv(DATA_DIR / "tsp_voxels.csv")
    align = pd.read_csv(DATA_DIR / "alignment_map.csv", parse_dates=["timestamp"])
    mon = pd.read_csv(DATA_DIR / "tbm_monitoring.csv", parse_dates=["timestamp"])

    profile = tsp.groupby("chainage_m", as_index=False).agg(
        q=("low_velocity_anomaly_score_q", "mean"),
        vp=("Vp_m_per_s", "mean"),
    )
    profile["x_rel"] = rel_chainage(profile["chainage_m"])

    fig = plt.figure(figsize=(10.8, 8.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.4, 1.2, 1.0], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    ax.plot(profile["x_rel"], profile["q"], color="#1f6f8b", lw=1.8, label="mean TSP anomaly score q_i")
    ax.fill_between(profile["x_rel"], 0, profile["q"], color="#9ecae1", alpha=0.35)
    ax.axvline(0, color="#9a3412", lw=1.2, ls="--", label="stuck event DyK1017+205")
    ax.axvspan(FAULT_START - EVENT_CHAINAGE, FAULT_END - EVENT_CHAINAGE, color="#d73027", alpha=0.12, label="F37 fault zone")
    ax.annotate("F37 start\n+72 m", xy=(FAULT_START - EVENT_CHAINAGE, 0.26), xytext=(50, 0.30), arrowprops=dict(arrowstyle="->", lw=0.8), ha="center")
    ax.annotate("TBM stuck event", xy=(0, 0.22), xytext=(-32, 0.29), arrowprops=dict(arrowstyle="->", lw=0.8), ha="center")
    ax.set_xlim(-50, 90)
    ax.set_ylim(0, max(profile["q"]) * 1.18)
    ax.set_xlabel("distance to confirmed event chainage (m)")
    ax.set_ylabel("mean q_i")
    ax.set_title("A. Longitudinal setting: event location, TSP anomalies and F37 fault influence")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)

    ax = fig.add_subplot(gs[1])
    section = tsp[(tsp["chainage_m"].between(EVENT_CHAINAGE - 6, EVENT_CHAINAGE + 2)) & (tsp["low_velocity_anomaly_score_q"] > 0.35)].copy()
    circle_outer = plt.Circle((0, 0), TBM_RADIUS + 1.0, facecolor="#f3efe7", edgecolor="#b89b72", lw=1.0, alpha=0.8, label="surrounding rock")
    circle_shield = plt.Circle((0, 0), TBM_RADIUS, facecolor="#eef3f7", edgecolor="#333333", lw=1.2, alpha=0.95, label="shield boundary")
    ax.add_patch(circle_outer)
    ax.add_patch(circle_shield)
    sc = ax.scatter(section["y_m"], section["z_m"], c=section["low_velocity_anomaly_score_q"], s=28, cmap="YlOrRd", vmin=0, vmax=1, edgecolor="none", alpha=0.85)
    ax.annotate("low-velocity\nvoxel cloud", xy=(0, 4.1), xytext=(-6.8, 6.4), arrowprops=dict(arrowstyle="->", lw=0.8), ha="center")
    ax.annotate("squeezing\nsurrounding rock", xy=(-3.8, 0), xytext=(-8.0, -4.8), arrowprops=dict(arrowstyle="->", lw=0.8), ha="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-9, 9)
    ax.set_ylim(-7, 7)
    ax.set_xlabel("local y (m)")
    ax.set_ylabel("local z (m)")
    ax.set_title("B. Cross-section: shield, surrounding rock and low-velocity voxels near the event")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="q_i")
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.5)

    ax = fig.add_subplot(gs[2])
    ax.plot(align["timestamp"], rel_chainage(align["face_chainage_m"]), color="#333333", lw=1.5, label="TBM face chainage")
    ax.axvspan(EVENT_START, EVENT_CONFIRM, color="#e8b45b", alpha=0.22, lw=0)
    ax.axvline(EVENT_CONFIRM, color="#9a3412", lw=1.2, ls="--")
    ax2 = ax.twinx()
    ax2.plot(mon["timestamp"], mon["shield_pressure_bar"], color="#b22222", lw=1.2, alpha=0.85, label="shield pressure")
    ax.set_ylabel("face x - event (m)")
    ax2.set_ylabel("shield pressure (bar)")
    ax.set_xlabel("monitoring timestamp")
    ax.set_title("C. Alignment: TSP chainage, TBM face chainage, monitoring time and event window")
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    savefig(fig, "fig1_spatial_setting_event_tsp_anomalies")


def compute_candidate_edges(face_chainage: float, top_n: int = 36, radius: float = 0.9) -> pd.DataFrame:
    tsp = pd.read_csv(DATA_DIR / "tsp_voxels.csv")
    nodes = pd.read_csv(DATA_DIR / "tbm_surface_nodes.csv")
    vox = tsp[tsp["low_velocity_anomaly_score_q"] > 0.05].copy()
    coords = vox[["chainage_m", "y_m", "z_m"]].to_numpy()
    tree = cKDTree(coords)

    node_coords = np.column_stack(
        [
            face_chainage + nodes["local_x_m"].to_numpy(),
            nodes["local_y_m"].to_numpy(),
            nodes["local_z_m"].to_numpy(),
        ]
    )
    records = []
    for node_idx, nearby in enumerate(tree.query_ball_point(node_coords, r=radius)):
        if not nearby:
            continue
        node = nodes.iloc[node_idx]
        node_xyz = node_coords[node_idx]
        for vi in nearby:
            voxel = vox.iloc[vi]
            vec = coords[vi] - node_xyz
            dist = float(np.linalg.norm(vec))
            if dist == 0:
                normal_comp = 1.0
            else:
                normal = np.array([node["normal_x"], node["normal_y"], node["normal_z"]], dtype=float)
                normal_comp = max(float(np.dot(vec / dist, normal)), 0.0)
            spatial_weight = np.exp(-(dist**2) / (2 * 0.55**2)) * (0.35 + 0.65 * normal_comp)
            q = float(voxel["low_velocity_anomaly_score_q"])
            contribution = spatial_weight * q
            records.append(
                {
                    "voxel_chainage_m": voxel["chainage_m"],
                    "voxel_y_m": voxel["y_m"],
                    "voxel_z_m": voxel["z_m"],
                    "tbm_node_chainage_m": node_xyz[0],
                    "tbm_node_y_m": node_xyz[1],
                    "tbm_node_z_m": node_xyz[2],
                    "component": node["component"],
                    "distance_m": dist,
                    "normal_compatibility": normal_comp,
                    "edge_weight": spatial_weight,
                    "anomaly_score_q": q,
                    "weighted_contribution": contribution,
                }
            )
    edges = pd.DataFrame(records)
    if edges.empty:
        return edges
    return edges.sort_values("weighted_contribution", ascending=False).head(top_n).reset_index(drop=True)


def figure_2_dynamic_snapshots() -> None:
    tsp = pd.read_csv(DATA_DIR / "tsp_voxels.csv")
    snapshots = [
        ("A. Before event development\n2024-08-03 05:00", pd.Timestamp("2024-08-03 05:00:00"), 1017197.111),
        ("B. Near-stuck window begins\n2024-08-03 13:00", pd.Timestamp("2024-08-03 13:00:00"), 1017204.800),
        ("C. Confirmed-near graph snapshot\n2024-08-03 16:20", pd.Timestamp("2024-08-03 16:20:00"), 1017204.990),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    low_vox = tsp[tsp["low_velocity_anomaly_score_q"] > 0.35].copy()
    low_vox["x_rel"] = rel_chainage(low_vox["chainage_m"])

    for ax, (title, _, face) in zip(axes, snapshots):
        area = low_vox[low_vox["x_rel"].between(rel_chainage(face) - 18, rel_chainage(face) + 8)]
        ax.scatter(area["x_rel"], area["z_m"], c=area["low_velocity_anomaly_score_q"], cmap="YlOrRd", vmin=0, vmax=1, s=12, alpha=0.25, edgecolor="none")
        add_tbm_side(ax, face, y0=-4, y1=4, alpha=0.78)
        edges = compute_candidate_edges(face, top_n=28)
        if not edges.empty:
            segments = []
            widths = []
            colors = []
            max_c = max(edges["weighted_contribution"].max(), 1e-6)
            for _, row in edges.iterrows():
                target_z = {"front_shield": 1.7, "middle_shield": 0.0, "tail_shield": -1.7, "cutterhead": 0.0}.get(row["component"], row["tbm_node_z_m"])
                segments.append(
                    [
                        (rel_chainage(row["voxel_chainage_m"]), row["voxel_z_m"]),
                        (rel_chainage(row["tbm_node_chainage_m"]), target_z),
                    ]
                )
                widths.append(0.4 + 3.0 * row["weighted_contribution"] / max_c)
                colors.append(COMPONENT_COLORS.get(row["component"], "#555555"))
            lc_shadow = LineCollection(segments, colors="#222222", linewidths=[w + 1.1 for w in widths], alpha=0.35)
            ax.add_collection(lc_shadow)
            lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.95)
            ax.add_collection(lc)
            ax.scatter(
                rel_chainage(edges["voxel_chainage_m"]),
                edges["voxel_z_m"],
                s=28 + 55 * edges["weighted_contribution"] / max_c,
                c=[COMPONENT_COLORS.get(c, "#555555") for c in edges["component"]],
                edgecolors="black",
                linewidths=0.35,
                alpha=0.95,
                zorder=5,
            )
            ax.text(
                0.03,
                0.05,
                f"top {len(edges)} candidate edges",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#aaaaaa", alpha=0.85),
            )
        ax.axvline(0, color="#9a3412", ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlim(rel_chainage(face) - 18, rel_chainage(face) + 8)
        ax.set_ylim(-6.4, 6.4)
        ax.set_xlabel("distance to event chainage (m)")
        ax.grid(True, color="#dddddd", lw=0.5, alpha=0.6)
    axes[0].set_ylabel("vertical coordinate z (m)")
    handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=comp.replace("_", " "))
        for comp, color in COMPONENT_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Dynamic rock-TBM spatial relations before and during shield sticking", y=0.98, fontsize=12)
    fig.subplots_adjust(bottom=0.22, top=0.82, wspace=0.18)
    savefig(fig, "fig2_dynamic_rock_tbm_spatial_relation_snapshots")


def figure_3_space_time_matrix() -> None:
    tsp = pd.read_csv(DATA_DIR / "tsp_voxels.csv")
    comp = pd.read_csv(DATA_DIR / "component_readout.csv", parse_dates=["timestamp"])
    mon = pd.read_csv(DATA_DIR / "tbm_monitoring.csv", parse_dates=["timestamp"])
    df = comp.merge(mon, on=["timestamp", "face_chainage_m"], how="inner")
    df["x_rel"] = rel_chainage(df["face_chainage_m"])

    profile = tsp.groupby("chainage_m", as_index=False)["low_velocity_anomaly_score_q"].mean()
    profile["x_rel"] = rel_chainage(profile["chainage_m"])

    fig = plt.figure(figsize=(10.8, 7.4))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.55, 1.35, 1.65, 0.35], hspace=0.18)

    x_min, x_max = -15.5, 0.3
    ax = fig.add_subplot(gs[0])
    band = profile[profile["x_rel"].between(x_min, x_max)]
    arr = band["low_velocity_anomaly_score_q"].to_numpy()[None, :]
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", extent=[band["x_rel"].min(), band["x_rel"].max(), 0, 1], vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_ylabel("TSP\nq_i")
    ax.set_title("A. TSP anomaly band")
    ax.axvspan(-0.2, 0, color="#e8b45b", alpha=0.28, lw=0)
    ax.axvline(0, color="#9a3412", lw=1.1, ls="--")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="q_i")

    ax = fig.add_subplot(gs[1], sharex=ax)
    components = ["cutterhead", "front_shield", "middle_shield", "tail_shield"]
    mat = df[components].T.to_numpy()
    im2 = ax.imshow(mat, aspect="auto", cmap="YlOrBr", extent=[df["x_rel"].min(), df["x_rel"].max(), len(components), 0], vmin=0, vmax=max(0.85, mat.max()))
    ax.set_yticks(np.arange(len(components)) + 0.5)
    ax.set_yticklabels([c.replace("_", " ") for c in components])
    ax.set_title("B. Component-level geological anomaly index I_c(t)")
    ax.axvspan(-0.2, 0, color="#e8b45b", alpha=0.28, lw=0)
    ax.axvline(0, color="#9a3412", lw=1.1, ls="--")
    fig.colorbar(im2, ax=ax, fraction=0.03, pad=0.01, label="I_c(t)")

    ax = fig.add_subplot(gs[2], sharex=ax)
    variables = [
        ("shield_pressure_bar", "pressure"),
        ("shield_displacement_mm", "displacement"),
        ("total_thrust_kN", "thrust"),
        ("advance_rate_m_per_h", "advance rate"),
        ("cutterhead_rpm", "RPM"),
    ]
    rows = []
    for col, _ in variables:
        v = df[col].to_numpy(dtype=float)
        rows.append((v - np.nanmin(v)) / max(np.nanmax(v) - np.nanmin(v), 1e-9))
    monitor_mat = np.vstack(rows)
    im3 = ax.imshow(monitor_mat, aspect="auto", cmap="RdYlBu_r", extent=[df["x_rel"].min(), df["x_rel"].max(), len(variables), 0], vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(variables)) + 0.5)
    ax.set_yticklabels([name for _, name in variables])
    ax.set_title("C. Monitoring response strips (normalized)")
    ax.axvspan(-0.2, 0, color="#e8b45b", alpha=0.28, lw=0)
    ax.axvline(0, color="#9a3412", lw=1.1, ls="--")
    fig.colorbar(im3, ax=ax, fraction=0.03, pad=0.01, label="normalized value")

    ax = fig.add_subplot(gs[3], sharex=ax)
    ax.set_ylim(0, 1)
    ax.axvspan(-0.2, 0, color="#e8b45b", alpha=0.35, lw=0)
    ax.axvline(0, color="#9a3412", lw=1.1, ls="--")
    ax.text(-0.1, 0.65, "near-stuck window", ha="center", va="center", color="#7c2d12")
    ax.text(0.05, 0.25, "confirmed\nstuck", ha="left", va="center", color="#7c2d12")
    ax.set_yticks([])
    ax.set_xlabel("TBM face distance to confirmed event chainage (m)")
    ax.set_title("D. Event annotation")
    ax.set_xlim(x_min, x_max)
    for upper in fig.axes[:-1]:
        if upper is not ax:
            plt.setp(upper.get_xticklabels(), visible=False)
    savefig(fig, "fig3_space_time_component_indices_monitoring")


def figure_4_edge_provenance() -> None:
    edges = pd.read_csv(DATA_DIR / "edge_provenance_top50.csv", parse_dates=["timestamp"])
    nodes = pd.read_csv(DATA_DIR / "tbm_surface_nodes.csv")
    top = edges.sort_values("weighted_contribution", ascending=False).head(8).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    face = float(top["face_chainage_m"].iloc[0])

    fig = plt.figure(figsize=(12.4, 7.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.15, 1.2], hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    shield_nodes = nodes[nodes["component"].isin(["front_shield", "middle_shield", "tail_shield"])].copy()
    shield_nodes["x_rel"] = rel_chainage(face + shield_nodes["local_x_m"])
    for comp, g in shield_nodes.groupby("component"):
        ax.scatter(g["x_rel"], g["local_y_m"], g["local_z_m"], s=8, color=COMPONENT_COLORS[comp], alpha=0.28, label=comp.replace("_", " "))
    for _, row in top.iterrows():
        node = nodes[nodes["tbm_node_id"] == row["tbm_node_id"]].iloc[0]
        nx = rel_chainage(face + node["local_x_m"])
        ny, nz = node["local_y_m"], node["local_z_m"]
        vx, vy, vz = rel_chainage(row["voxel_chainage_m"]), row["voxel_y_m"], row["voxel_z_m"]
        color = COMPONENT_COLORS.get(row["component"], "#555555")
        ax.plot([vx, nx], [vy, ny], [vz, nz], color=color, lw=0.8 + 3.0 * row["weighted_contribution"] / top["weighted_contribution"].max(), alpha=0.9)
        ax.scatter([vx], [vy], [vz], color="#d73027", s=38, edgecolor="black", linewidth=0.3)
        ax.text(vx, vy, vz + 0.25, str(int(row["rank"])), fontsize=8)
    ax.set_xlabel("x-event (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("A. Local 3D rock-shield edge provenance")
    ax.view_init(elev=18, azim=-62)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02))

    ax = fig.add_subplot(gs[0, 1])
    add_tbm_side(ax, face, y0=-4, y1=4, alpha=0.40)
    max_c = top["weighted_contribution"].max()
    for _, row in top.iterrows():
        node = nodes[nodes["tbm_node_id"] == row["tbm_node_id"]].iloc[0]
        nx = rel_chainage(face + node["local_x_m"])
        nz = node["local_z_m"]
        vx, vz = rel_chainage(row["voxel_chainage_m"]), row["voxel_z_m"]
        color = COMPONENT_COLORS.get(row["component"], "#555555")
        visible_target_z = {"front_shield": 1.7, "middle_shield": 0.0, "tail_shield": -1.7, "cutterhead": 0.0}.get(row["component"], nz)
        ax.plot([vx, nx], [vz, visible_target_z], color="#222222", lw=2.4, alpha=0.22)
        ax.plot([vx, nx], [vz, visible_target_z], color=color, lw=0.9 + 3.2 * row["weighted_contribution"] / max_c, alpha=0.9)
        ax.scatter(vx, vz, color="#d73027", s=44, edgecolor="black", linewidth=0.3)
        ax.text(vx + 0.1, vz + 0.1, str(int(row["rank"])), fontsize=8)
    ax.axvline(0, color="#9a3412", lw=1.0, ls="--")
    ax.set_xlim(-11, 1.5)
    ax.set_ylim(-5.4, 5.4)
    ax.set_xlabel("distance to event chainage (m)")
    ax.set_ylabel("vertical coordinate z (m)")
    ax.set_title("B. Longitudinal section of top contributing rock-shield edges")
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.65)

    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")
    display = top[
        [
            "rank",
            "component",
            "voxel_chainage_m",
            "distance_m",
            "normal_compatibility",
            "anomaly_score_q",
            "edge_weight",
            "weighted_contribution",
        ]
    ].copy()
    display["x_rel_m"] = display["voxel_chainage_m"] - EVENT_CHAINAGE
    display = display[["rank", "component", "x_rel_m", "distance_m", "normal_compatibility", "anomaly_score_q", "edge_weight", "weighted_contribution"]]
    display["component"] = display["component"].str.replace("_", " ", regex=False)
    for col in ["x_rel_m", "distance_m", "normal_compatibility", "anomaly_score_q", "edge_weight", "weighted_contribution"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    display.to_csv(TABLE_DIR / "figure4_top8_edge_provenance.csv", index=False)
    # Keep the old output name for downstream compatibility.
    display.to_csv(TABLE_DIR / "figure4_top10_edge_provenance.csv", index=False)
    table = table_ax.table(
        cellText=display.values,
        colLabels=["Rank", "Component", "Rel. chainage", "Distance", "Normal", "q_i", "w_ij(t)", "Contribution"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.06, 0.18, 0.12, 0.10, 0.10, 0.08, 0.10, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.28)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    table_ax.set_title("C. Top edge-level provenance records near the stuck state", pad=8)
    savefig(fig, "fig4_edge_level_spatial_provenance")


def main() -> None:
    setup()
    build_table_1()
    figure_1_spatial_setting()
    figure_2_dynamic_snapshots()
    figure_3_space_time_matrix()
    figure_4_edge_provenance()
    print(f"Generated figures in {FIG_DIR}")
    print(f"Generated tables in {TABLE_DIR}")


if __name__ == "__main__":
    main()
