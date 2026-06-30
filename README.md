# Rock--TBM Spatial Interaction Descriptor Project

This repository contains experiment code and IJGIS manuscript materials for a
geometry-constrained, chainage-referenced rock--TBM spatial interaction graph
framework.

The current method represents TSP-derived rock voxels and parameterized TBM
surface components as heterogeneous spatial entities, constructs
geometry-constrained candidate rock--machine relations, and aggregates those
relations into component--chainage descriptors:

- `A_c(t)`: component-level geometric exposure;
- `I_c(t)`: geometry-weighted TSP anomaly intensity;
- `e_{t+h}`: persistence residual of TBM monitoring response.

The paper's main diagnostic test is whether `I_c(t)` co-varies with persistence
residuals, not whether a deep model beats persistence in absolute forecasting.

## Main Cases

| Case | Tunnel | Start chainage | Primary role |
|---|---|---|---|
| `bsll_dyk1017_205` | BSLL | DyK1017+205 | compact h=1 descriptor-residual case |
| `bsll_dyk1017_205_h3` | BSLL | DyK1017+205 | compact h=3 residual-sensitivity case |
| `sjls_dyk1252_411` | SJLS | Dyk1252+411 | strongest external TSP descriptor evidence |

## Repository Layout

```text
experiments/
  config/       semantic case YAML files
  data/         raw and processed TSP/TBM monitoring inputs
  scripts/      descriptor collection, sensitivity, and figure scripts
  src/          reusable data, graph, diagnostics, and visualization code
  outputs/      generated descriptor tables and figures
event_centered_experiment_figures/
  figures/      reusable event-centred manuscript figures
  tables/       event-centred figure/table source summaries
paper/
  new-ijgis/                active revised manuscript
tbm_shield_sticking/        event-alignment and edge-provenance evidence package
```

## Run From `experiments/`

Descriptor evidence:

```powershell
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205.yaml --output-dir outputs/descriptors/bsll_dyk1017_205
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205_h3.yaml --output-dir outputs/descriptors/bsll_dyk1017_205_h3
python scripts/collect_spatial_descriptors.py --config config/sjls_dyk1252_411.yaml --output-dir outputs/descriptors/sjls_dyk1252_411
python scripts/summarize_spatial_descriptors.py
```

Threshold sensitivity and figures:

```powershell
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205.yaml
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_descriptor_sensitivity.py --config config/sjls_dyk1252_411.yaml
python scripts/make_descriptor_figures.py
```

## Current Paper Positioning

The active manuscript is `paper/new-ijgis/revised_tex_project_v2/main_revised.tex`.
The previous `paper/ijgis-template/` draft has been removed after its usable
bibliography, DOI audit, citation anchors, and figures were migrated into the
active manuscript.

The defensible contribution is:

- a chainage-referenced rock--TBM spatial entity model;
- a geometry-constrained rock--machine candidate relation graph;
- component--chainage spatial interaction descriptors evaluated against
  persistence residuals.

Do not frame the paper as a prediction-superiority or contact-force recovery
study. The descriptor maps are geometry-weighted TSP anomaly summaries over
candidate rock--machine relations; they are not measured contact pressure,
jamming probability, or calibrated risk.
