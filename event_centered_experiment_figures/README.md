# Event-centred Experiment Figures

This folder contains the spatial event-profile figure system for the IJGIS-style
event-centred evaluation of the dynamic rock-TBM spatial relation graph.

The figures are generated from the real shield-sticking event dataset in
`../tbm_shield_sticking/`. The purpose is not to prove a general shield-sticking
mechanism or build a prediction benchmark. The purpose is to show whether the
graph converts TSP-derived geological anomalies into component-level geological
anomaly indices and makes the shield-sticking interpretation traceable to
specific rock-TBM spatial relations.

## Experiment Logic

The experiment is organized around three questions:

1. Are the event record, TSP field, TBM monitoring, TBM geometry, and graph snapshots aligned in a common time-chainage coordinate system?
2. Does the dynamic rock-TBM spatial relation graph convert TSP anomaly scores `q_i` into component-level geological anomaly indices `I_c(t)` through candidate spatial relations `w_ij(t)`?
3. Does the shield-group geological anomaly index `I_shield(t)` provide a more spatially explicit event interpretation than global or pooled summaries, and can it be traced back to top contributing rock-shield edges?

## Generated Outputs

`tables/table1_case_data_alignment_summary.md`

: Table 1. Case event, input data and alignment used for event-centred evaluation.

`figures/fig1_spatial_setting_event_tsp_anomalies.png`

: Figure 1. Spatial setting of the shield-sticking event and TSP-derived geological anomalies. This figure shows the longitudinal event setting, local TBM cross-section, low-velocity voxel cloud, and time-chainage-monitoring alignment.

`figures/fig2_dynamic_rock_tbm_spatial_relation_snapshots.png`

: Figure 2. Dynamic rock-TBM spatial relations before and during shield sticking. This is the main spatial relation graph figure, showing TBM movement, low-velocity voxels, component-labelled TBM segments, and top candidate rock-TBM spatial relations.

`figures/fig3_space_time_component_indices_monitoring.png`

: Figure 3. Space-time evolution of component-level geological anomaly indices and monitoring responses. This figure replaces ordinary time-series plots with aligned anomaly bands, component-by-chainage heatmaps, and monitoring response strips.

`figures/fig4_edge_level_spatial_provenance.png`

: Figure 4. Edge-level provenance of component-level geological anomaly index near the stuck state. This figure traces the interpretation back to low-velocity rock voxels, shield surface nodes, candidate edges, and edge attributes.

PDF versions are also generated for manuscript integration.

## Data Sources

- `../tbm_shield_sticking/event_metadata.csv`
- `../tbm_shield_sticking/alignment_map.csv`
- `../tbm_shield_sticking/tsp_voxels.csv`
- `../tbm_shield_sticking/tbm_surface_nodes.csv`
- `../tbm_shield_sticking/component_readout.csv`
- `../tbm_shield_sticking/tbm_monitoring.csv`
- `../tbm_shield_sticking/event_vs_reference_summary.csv`
- `../tbm_shield_sticking/edge_provenance_top50.csv`
- `../tbm_shield_sticking/graph_edges_event_sample_top500.csv`

## Reproduce

Run from the repository root:

```powershell
python event_centered_experiment_figures\generate_event_centered_figures.py
```

The script writes PNG/PDF figures to `figures/` and table outputs to `tables/`.

## Figure Design Boundary

The main figure system must remain spatial. Avoid reverting Figure 2 or Figure 3
to ordinary monitoring time series. Monitoring responses are supporting evidence
attached to the spatial relation interpretation, not the centre of the visual
argument.

## Manuscript Boundary

Do not expand this experiment into AUC/F1 metrics, large baseline comparisons,
contact-force inversion, shield-sticking risk probability maps, or detailed
engineering mitigation workflows. The manuscript argument should stay on the
GIScience contribution: preserving component-level rock-TBM spatial relations
and providing edge-level provenance for event-centred interpretation.
