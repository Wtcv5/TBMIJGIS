# TBM Shield-Sticking Event Dataset

## Purpose

This dataset contains real monitoring, TSP, geometry, graph-edge, and validation
data for the TBM shield-sticking event.

It is anchored to the reported facts of a shield jamming incident on the right
line of the Boshula Ridge Tunnel:

- The shield jamming incident was confirmed to have occurred on August 3, 2024, at 16:30.
- The shield pressure rose from a normal value of approximately 100 bar to a peak of approximately 362 bar.
- The shield displacement reached approximately -73 mm, approaching the limit of -80 mm.
- The total thrust increased to approximately 45,000 kN.
- Between 13:00 and 16:30, the cutterhead continued to rotate, but the tunneling speed was almost zero.
- The event mileage is anchored at DyK1017+205, approximately 72 m from the starting point of the F37 fault intersection.

## Core Experimental Logic

This dataset supports an event-centric chain of evidence:

1. `tsp_voxels.csv` provides the low-velocity, weak strata field around the tunnel.
2. `tbm_surface_nodes.csv` provides TBM surface nodes with component labels.
3. `component_readout.csv` converts the TSP field into geological exposure trajectories with component indices.
4. `tbm_monitoring.csv` provides TBM response trajectories for the event.
5. `edge_provenance_top50.csv` provides the top 50 rock-shield candidate edges that contributed most to the near-stuck event.

## Files

- `event_metadata.csv`: Event anchors and reporting facts used for validation.
- `alignment_map.csv`: Timestamps, surface mileage, event mileage, and event window flags.
- `tsp_voxels.csv`: TSP-derived rock voxels containing P-wave velocity (Vp), S-wave velocity (Vs), and low-velocity anomaly score q.
- `tbm_surface_nodes.csv`: TBM surface nodes annotated in the local machine coordinate system.
- `component_readout.csv`: Cutterhead/front shield/middle shield/tail shield readings, shield group readings, global anomalies, and summary readings.
- `tbm_monitoring.csv`: Shield pressure, shield displacement, thrust, advance velocity, and cutterhead rotation speed.
- `graph_edges_event_sample_top500.csv`: Top 500 candidate rock-machine edges as of 16:20 on August 3, 2024.
- `edge_provenance_top50.csv`: Top 50 edge-level provenance records within the same event time period.
- `event_vs_reference_summary.csv`: Comparison summary of the event window and the reference window.
- `validation_checks.csv`: Basic validation metrics for the event data package.
- `DATA_REQUEST_CHECKLIST.md`: Field checklist for auditing completeness and provenance.
- `plot_event_alignment.py`: Script for plotting event alignment.
- `plot_edge_provenance.py`: Script for edge provenance plotting.

## Important Usage Rules

Use this dataset as real event validation data:

- Keep the field names stable when running scripts.
- Preserve the event-centric alignment between TSP, TBM monitoring, machine geometry, and event logs.
- Keep the event-centric evidence logic unchanged.
