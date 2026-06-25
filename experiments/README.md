# Experiments

This directory contains the runnable pipeline for the explicit rock--TBM spatial
interaction descriptor framework.

## Case Configs

| Config | Purpose |
|---|---|
| `config/bsll_dyk1017_205.yaml` | BSLL, DyK1017+205, h=1 descriptor-residual case |
| `config/bsll_dyk1017_205_h3.yaml` | BSLL, DyK1017+205, h=3 descriptor-residual case |
| `config/sjls_dyk1252_411.yaml` | SJLS, Dyk1252+411, external TSP descriptor case |
| `config/*_quick.yaml` | short smoke-test versions |

## Core Commands

```powershell
# Explicit component-chainage descriptors
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205.yaml --output-dir outputs/descriptors/bsll_dyk1017_205
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205_h3.yaml --output-dir outputs/descriptors/bsll_dyk1017_205_h3
python scripts/collect_spatial_descriptors.py --config config/sjls_dyk1252_411.yaml --output-dir outputs/descriptors/sjls_dyk1252_411

# Cross-case summaries
python scripts/summarize_spatial_descriptors.py

# Graph-threshold sensitivity
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205.yaml
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_descriptor_sensitivity.py --config config/sjls_dyk1252_411.yaml

# Publication figures for the descriptor evidence chain
python scripts/make_descriptor_figures.py
```

## Source Modules

```text
src/data/           TSP loading, monitoring preprocessing, geometry alignment
src/graph/          node, edge, and graph-sequence construction
src/diagnostics/    explicit A_c(t), I_c(t), residual, and association logic
src/visualization/  IJGIS-style plotting helpers
```

## Main Outputs

```text
outputs/descriptors/<case>/component_spatial_descriptors.csv
outputs/descriptors/<case>/descriptor_residual_association.csv
outputs/descriptors/<case>/graph_construction_summary.csv
outputs/descriptors/<case>/sensitivity/
outputs/descriptors/descriptor_case_summary.csv
outputs/descriptors/descriptor_association_all.csv
outputs/figures/fig1_method_framework.*
outputs/figures/fig6_descriptor_evidence.*
outputs/figures/fig7_descriptor_sensitivity.*
```

The descriptor values are geometry-weighted TSP anomaly summaries over screened
candidate rock--machine relations. They are not contact force, contact pressure,
or calibrated jamming risk.
