# Event-centred Experiment Figures

This folder contains the minimal experiment figure system for the IJGIS-style
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

`figures/fig1_event_centred_workflow_alignment.png`

: Figure 1. Event-centred evaluation workflow and data alignment. This is the bridge between data objects, graph relations, and event interpretation.

`figures/fig2_event_aligned_indices_monitoring.png`

: Figure 2. Event-aligned geological anomaly indices and TBM monitoring responses. This is the main result figure.

`figures/fig3_component_indexing_added_value.png`

: Figure 3. Added value of component-level indexing compared with global and pooled summaries.

`figures/fig4_edge_level_provenance.png`

: Figure 4. Edge-level provenance of the shield-sticking interpretation.

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

## Reproduce

Run from the repository root:

```powershell
python event_centered_experiment_figures\generate_event_centered_figures.py
```

The script writes figures to `figures/` and table outputs to `tables/`.

## Manuscript Boundary

Do not expand this experiment into AUC/F1 metrics, large baseline comparisons,
contact-force inversion, shield-sticking risk probability maps, or detailed
engineering mitigation workflows. The manuscript argument should stay on the
GIScience contribution: preserving component-level rock-TBM spatial relations
and providing edge-level provenance for event-centred interpretation.
