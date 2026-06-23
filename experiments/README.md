# Experiments

This directory contains the runnable pipeline for the rock-TBM interaction graph
sequence framework.

## Case Configs

| Config | Purpose |
|---|---|
| `config/bsll_dyk1017_205.yaml` | BSLL, DyK1017+205, one-step response prediction |
| `config/bsll_dyk1017_205_h3.yaml` | BSLL, DyK1017+205, three-step advance prediction |
| `config/sjls_dyk1252_411.yaml` | SJLS, Dyk1252+411, external Vp/Vs TSP case |
| `config/*_quick.yaml` | short smoke-test versions with reduced training cost |

## Core Commands

```powershell
# Formal graph-sequence experiments
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205.yaml
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_graph_sequence_case.py --config config/sjls_dyk1252_411.yaml

# Case-level metrics summary
python scripts/summarize_case_results.py

# Post-hoc interpretation evidence
python scripts/collect_evidence.py --config config/bsll_dyk1017_205.yaml --run-dir outputs/bsll_dyk1017_205 --output-dir outputs/evidence/bsll_dyk1017_205
python scripts/collect_evidence.py --config config/bsll_dyk1017_205_h3.yaml --run-dir outputs/bsll_dyk1017_205_h3 --output-dir outputs/evidence/bsll_dyk1017_205_h3
python scripts/collect_evidence.py --config config/sjls_dyk1252_411.yaml --run-dir outputs/sjls_dyk1252_411 --output-dir outputs/evidence/sjls_dyk1252_411
python scripts/summarize_interpretation_evidence.py
```

## Source Modules

```text
src/data/           TSP loading, monitoring preprocessing, geometry alignment
src/graph/          node, edge, and graph-sequence construction
src/models/         baselines, GNN encoder, graph-sequence models
src/training/       datasets, losses, training loops, metrics
src/visualization/  graph, prediction, hotspot, and IJGIS-style plotting
```

## Output Policy

Generated run directories under `outputs/` are reproducible artifacts. The
formal run names are:

- `outputs/bsll_dyk1017_205/`
- `outputs/bsll_dyk1017_205_h3/`
- `outputs/sjls_dyk1252_411/`
- `outputs/evidence/`
