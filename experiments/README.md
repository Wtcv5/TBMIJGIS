# Experiments

This directory contains the runnable evidence pipeline for the rock--TBM spatial relation graph manuscript. The current pipeline is descriptor-based: it constructs geometry-constrained rock--TBM candidate relations and reports component-level geological anomaly descriptors. It does not train or require prediction checkpoints.

## Layout

```text
config/       case configurations for formal and smoke-test runs
data/         retained input data and processed voxel/monitoring tables
outputs/      generated outputs; only README.md is kept in the clean project
scripts/      command-line experiment and figure-generation entry points
src/          reusable data, graph, diagnostics, and visualization modules
```

## Data Kept

```text
data/raw/
  tsp.csv
  monitoring.csv
data/processed/sjls_dyk1252_411/
  tsp_sjls_dyk1252_411.csv
  monitoring_sjls_dyk1252_411.csv
  geology_drivers_sjls_dyk1252_411.csv
  tsp_sjls_dyk1252_411_audit.json
```

The large external SJLS raw text files were removed after conversion. The processed voxel table and audit JSON are the retained reproducible support files.

## Case Configs

| Config | Purpose |
|---|---|
| `config/bsll_dyk1017_205.yaml` | BSLL, DyK1017+205, h=1 descriptor-residual case |
| `config/bsll_dyk1017_205_h3.yaml` | BSLL, DyK1017+205, h=3 descriptor-residual case |
| `config/sjls_dyk1252_411.yaml` | SJLS, Dyk1252+411, processed external TSP descriptor case |
| `config/*_quick.yaml` | short smoke-test versions |

## Core Commands

Run these from `experiments/`.

```powershell
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205.yaml --output-dir outputs/descriptors/bsll_dyk1017_205
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205_h3.yaml --output-dir outputs/descriptors/bsll_dyk1017_205_h3
python scripts/collect_spatial_descriptors.py --config config/sjls_dyk1252_411.yaml --output-dir outputs/descriptors/sjls_dyk1252_411
python scripts/summarize_spatial_descriptors.py
```

Sensitivity and figures:

```powershell
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205.yaml
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_descriptor_sensitivity.py --config config/sjls_dyk1252_411.yaml
python scripts/make_descriptor_figures.py
```

Useful figure-specific commands:

```powershell
python scripts/gen_graph_construction_fig.py
python scripts/make_graph_publication_figure.py --config config/bsll_dyk1017_205.yaml
```

## Source Modules

```text
src/data/           TSP loading, monitoring preprocessing, geometry alignment
src/graph/          node, edge, and graph-sequence construction
src/diagnostics/    A_c(t), I_c(t), residual, association, and evidence logic
src/visualization/  manuscript-style plotting helpers
src/training/       lightweight sequence dataset wrapper used by diagnostics
```

## Generated Outputs

Generated outputs are intentionally not retained in the cleaned project. Recreate them with the commands above:

```text
outputs/descriptors/<case>/component_spatial_descriptors.csv
outputs/descriptors/<case>/descriptor_residual_association.csv
outputs/descriptors/<case>/graph_construction_summary.csv
outputs/descriptors/<case>/sensitivity/
outputs/descriptors/descriptor_case_summary.csv
outputs/descriptors/descriptor_association_all.csv
outputs/figures/
```

Descriptor values are geometry-weighted TSP anomaly summaries over screened candidate rock--machine relations. They are not contact force, contact pressure, jamming probability, or calibrated risk.
