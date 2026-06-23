# Rock-TBM Interaction Graph Sequence Project

This repository contains the experiment code and IJGIS manuscript materials for
a spatially explicit graph-sequence framework for rock-TBM interaction
interpretation.

The method represents TSP-derived rock voxels and parameterized TBM surface
nodes as a heterogeneous spatial graph sequence. Geometry-constrained
rock-machine edges define plausible interaction relations, and TBM monitoring
responses supervise prediction and attention-derived surface relevance.

## Main Cases

| Case | Tunnel | Start chainage | Primary role |
|---|---|---|---|
| `bsll_dyk1017_205` | BSLL | DyK1017+205 | one-step prediction and interpretation reference |
| `bsll_dyk1017_205_h3` | BSLL | DyK1017+205 | three-step advance-prediction sensitivity case |
| `sjls_dyk1252_411` | SJLS | Dyk1252+411 | main external TSP case for prediction consistency and geometry ablation |

Quick configs are provided only for smoke tests:

- `experiments/config/bsll_dyk1017_205_quick.yaml`
- `experiments/config/bsll_dyk1017_205_h3_quick.yaml`
- `experiments/config/sjls_dyk1252_411_quick.yaml`

## Repository Layout

```text
experiments/
  config/       semantic case YAML files
  data/         raw and processed TSP/TBM monitoring inputs
  scripts/      runnable pipelines and evidence collectors
  src/          reusable data, graph, model, training, and visualization code
  outputs/      generated results; mostly ignored by git except summaries/evidence
paper/
  ijgis-template/       LaTeX manuscript
  rewriting_output/     planning, evidence, and revision artifacts
IJGIS-papaers/          local target-journal and related-paper PDFs
docs/                   project-structure notes
```

## Run From `experiments/`

Smoke tests:

```powershell
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205_quick.yaml
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205_h3_quick.yaml
python scripts/run_graph_sequence_case.py --config config/sjls_dyk1252_411_quick.yaml
```

Formal runs:

```powershell
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205.yaml
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_graph_sequence_case.py --config config/sjls_dyk1252_411.yaml
```

Post-hoc evidence and summaries:

```powershell
python scripts/collect_evidence.py --config config/sjls_dyk1252_411.yaml --run-dir outputs/sjls_dyk1252_411 --output-dir outputs/evidence/sjls_dyk1252_411
python scripts/summarize_case_results.py
python scripts/summarize_interpretation_evidence.py
```

## Current Paper Positioning

The defensible contribution is not broad, significant improvement over
Persistence on every dataset. The stronger argument is that the framework turns
TBM response modelling into a spatially explicit interaction-interpretation
problem:

- prediction metrics check response consistency;
- geometry ablations test whether rock-machine relations matter;
- attention-derived TBM surface hotspots provide interpretable spatial evidence;
- BSLL and SJLS show the same representation applied across different tunnel
  cases.

See `paper/rewriting_output/interpretation_evidence_bank.md` for the current
claim boundaries and traceable evidence.
