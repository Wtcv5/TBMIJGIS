from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tbm_shield_sticking"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

EVENT_START = pd.Timestamp("2024-08-03 13:00:00")
EVENT_CONFIRM = pd.Timestamp("2024-08-03 16:30:00")
EVENT_CHAINAGE = 1017205.0


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


def mark_event(ax: plt.Axes, x_is_time: bool = True) -> None:
    if x_is_time:
        ax.axvspan(EVENT_START, EVENT_CONFIRM, color="#e8b45b", alpha=0.18, lw=0)
        ax.axvline(EVENT_CONFIRM, color="#9a3412", lw=1.2, ls="--")
    else:
        ax.axvspan(-20, 0, color="#e8b45b", alpha=0.18, lw=0)
        ax.axvline(0, color="#9a3412", lw=1.2, ls="--")


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


def figure_1_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 2.45, 1.9, 0.9, "TSP geological field\nrock voxels + Vp/Vs\nanomaly score q_i"),
        (2.75, 2.45, 2.0, 0.9, "TBM geometry and\nexcavation step\ncomponent-labelled nodes"),
        (5.3, 2.45, 2.1, 0.9, "Dynamic rock-TBM\nspatial relation graph\ncandidate edges + w_ij(t)"),
        (7.95, 2.45, 1.8, 0.9, "Event-centred\ninterpretation\nI_c(t), monitoring,\nedge provenance"),
    ]
    colors = ["#d8eef2", "#e7ead1", "#eee1c7", "#eadde8"]
    for (x, y, w, h, label), color in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", linespacing=1.25)

    for x0, x1 in [(2.2, 2.75), (4.75, 5.3), (7.4, 7.95)]:
        ax.annotate("", xy=(x1, 2.9), xytext=(x0, 2.9), arrowprops=dict(arrowstyle="->", lw=1.0))

    ax.plot([0.6, 9.4], [1.2, 1.2], color="#333333", lw=1.1)
    ax.text(5.0, 0.82, "common excavation coordinate: time / chainage", ha="center")
    ax.axvspan(5.7, 7.9, ymin=0.22, ymax=0.34, color="#e8b45b", alpha=0.25)
    ax.plot([7.9, 7.9], [1.05, 1.35], color="#9a3412", ls="--", lw=1.2)
    ax.text(6.75, 1.45, "event-development window", ha="center", color="#7c2d12")
    ax.text(8.05, 1.72, "confirmed\nstuck time", ha="left", va="center", color="#7c2d12")

    ax.text(
        5.0,
        0.25,
        "Geological anomalies, TBM components and monitoring responses are aligned in a common excavation coordinate system.",
        ha="center",
        fontsize=9,
    )
    savefig(fig, "fig1_event_centred_workflow_alignment")


def figure_2_event_aligned() -> None:
    tsp = pd.read_csv(DATA_DIR / "tsp_voxels.csv")
    comp = pd.read_csv(DATA_DIR / "component_readout.csv", parse_dates=["timestamp"])
    mon = pd.read_csv(DATA_DIR / "tbm_monitoring.csv", parse_dates=["timestamp"])
    df = comp.merge(mon, on=["timestamp", "face_chainage_m"], how="inner")
    profile = tsp.groupby("chainage_m", as_index=False)["low_velocity_anomaly_score_q"].mean()
    profile["distance_to_event_m"] = profile["chainage_m"] - EVENT_CHAINAGE

    fig, axes = plt.subplots(4, 1, figsize=(9.5, 8.4), sharex=False)
    axes[0].plot(profile["distance_to_event_m"], profile["low_velocity_anomaly_score_q"], color="#1f6f8b", lw=1.6)
    axes[0].set_ylabel("mean q_i")
    axes[0].set_title("A. TSP geological anomaly along chainage")
    axes[0].set_xlabel("distance to confirmed event chainage (m)")
    mark_event(axes[0], x_is_time=False)

    axes[1].plot(df["timestamp"], df["cutterhead"], color="#8a8a8a", lw=1.0, label="cutterhead")
    axes[1].plot(df["timestamp"], df["front_shield"], color="#59a14f", lw=1.0, label="front shield")
    axes[1].plot(df["timestamp"], df["middle_shield"], color="#f28e2b", lw=1.2, label="middle shield")
    axes[1].plot(df["timestamp"], df["tail_shield"], color="#4e79a7", lw=1.0, label="tail shield")
    axes[1].plot(df["timestamp"], df["shield_group_readout"], color="#111111", lw=2.0, label="shield group")
    axes[1].set_ylabel("I_c(t)")
    axes[1].set_title("B. Component-level geological anomaly indices")
    axes[1].legend(ncol=5, loc="upper left")
    mark_event(axes[1])

    ax = axes[2]
    ax.plot(df["timestamp"], df["shield_pressure_bar"], color="#b22222", lw=1.6, label="shield pressure")
    ax.set_ylabel("pressure (bar)")
    ax2 = ax.twinx()
    ax2.plot(df["timestamp"], df["shield_displacement_mm"], color="#3b5b92", lw=1.4, label="shield displacement")
    ax2.set_ylabel("displacement (mm)")
    ax.set_title("C. TBM monitoring response")
    mark_event(ax)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    axes[3].plot(df["timestamp"], df["advance_rate_m_per_h"], color="#2f7d32", lw=1.6, label="advance rate")
    axes[3].plot(df["timestamp"], df["cutterhead_rpm"], color="#777777", lw=1.1, label="cutterhead RPM")
    axes[3].set_ylabel("rate / RPM")
    axes[3].set_title("D. Event window: advance approaches zero while cutterhead rotates")
    axes[3].legend(loc="upper right")
    mark_event(axes[3])

    for ax in axes:
        ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
    axes[1].tick_params(labelbottom=False)
    axes[2].tick_params(labelbottom=False)
    for label in axes[3].get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    axes[-1].set_xlabel("timestamp")
    fig.subplots_adjust(hspace=0.62)
    savefig(fig, "fig2_event_aligned_indices_monitoring")


def figure_3_added_value() -> None:
    comp = pd.read_csv(DATA_DIR / "component_readout.csv", parse_dates=["timestamp"])
    summary = pd.read_csv(DATA_DIR / "event_vs_reference_summary.csv")

    fig = plt.figure(figsize=(10.5, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.5, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    configs = [
        ("global_active_zone_anomaly", "A. Global geological anomaly summary", "#1f6f8b"),
        ("pooled_tbm_readout", "B. Pooled TBM anomaly summary", "#7a5195"),
        ("shield_group_readout", "C. Component-level shield index", "#111111"),
    ]
    for ax, (col, title, color) in zip(axes, configs):
        ax.plot(comp["timestamp"], comp[col], color=color, lw=1.8)
        mark_event(ax)
        ax.set_title(title)
        ax.set_ylabel("index")
        ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)

    axes[2].plot(comp["timestamp"], comp["front_shield"], color="#59a14f", lw=0.9, alpha=0.75, label="front")
    axes[2].plot(comp["timestamp"], comp["middle_shield"], color="#f28e2b", lw=0.9, alpha=0.75, label="middle")
    axes[2].plot(comp["timestamp"], comp["tail_shield"], color="#4e79a7", lw=0.9, alpha=0.75, label="tail")
    axes[2].legend(loc="upper left")

    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")
    rows = [
        ["Global summary", "No", "No", "weak ground exists"],
        ["Pooled TBM summary", "No", "Partly", "TBM is near anomalous ground"],
        ["Component-level index", "Yes", "Yes", "anomaly is related to shield components"],
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=["Summary type", "Component label", "Edge provenance", "Interpretation"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.26, 0.17, 0.17, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")

    selected = summary[summary["variable"].isin([c[0] for c in configs])].copy()
    selected.to_csv(TABLE_DIR / "figure3_summary_event_reference_values.csv", index=False)
    fig.autofmt_xdate()
    savefig(fig, "fig3_component_indexing_added_value")


def figure_4_edge_provenance() -> None:
    edges = pd.read_csv(DATA_DIR / "edge_provenance_top50.csv", parse_dates=["timestamp"])
    top = edges.sort_values("weighted_contribution", ascending=False).head(10).copy()
    top["rank"] = np.arange(1, len(top) + 1)

    fig = plt.figure(figsize=(12.2, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    colors = {"front_shield": "#59a14f", "middle_shield": "#f28e2b", "tail_shield": "#4e79a7", "cutterhead": "#8a8a8a"}
    for comp, g in top.groupby("component"):
        ax.scatter(
            g["voxel_y_m"],
            g["voxel_z_m"],
            s=60 + 260 * g["weighted_contribution"] / top["weighted_contribution"].max(),
            color=colors.get(comp, "#777777"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=comp.replace("_", " "),
        )
        for _, row in g.iterrows():
            ax.text(row["voxel_y_m"] + 0.08, row["voxel_z_m"] + 0.08, str(int(row["rank"])), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("voxel y (m)")
    ax.set_ylabel("voxel z (m)")
    ax.set_title("A. Top low-velocity voxels linked to shield components")
    ax.legend(loc="best")
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)

    table_ax = fig.add_subplot(gs[0, 1])
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
    display = display[
        [
            "rank",
            "component",
            "x_rel_m",
            "distance_m",
            "normal_compatibility",
            "anomaly_score_q",
            "edge_weight",
            "weighted_contribution",
        ]
    ]
    display["component"] = display["component"].str.replace("_", " ", regex=False)
    for col in ["x_rel_m", "distance_m", "normal_compatibility", "anomaly_score_q", "edge_weight", "weighted_contribution"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    display.to_csv(TABLE_DIR / "figure4_top10_edge_provenance.csv", index=False)
    table = table_ax.table(
        cellText=display.values,
        colLabels=["Rank", "Component", "x-event (m)", "Dist.", "Normal", "q_i", "w_ij(t)", "Contribution"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.06, 0.18, 0.12, 0.10, 0.10, 0.08, 0.10, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1, 1.22)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    table_ax.set_title("B. Edge-level provenance records", pad=12)
    savefig(fig, "fig4_edge_level_provenance")


def main() -> None:
    setup()
    build_table_1()
    figure_1_workflow()
    figure_2_event_aligned()
    figure_3_added_value()
    figure_4_edge_provenance()
    print(f"Generated figures in {FIG_DIR}")
    print(f"Generated tables in {TABLE_DIR}")


if __name__ == "__main__":
    main()
