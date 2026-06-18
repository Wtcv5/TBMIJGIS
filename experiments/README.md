# Experiments

This directory contains the runnable experiment pipeline for the rock-TBM
interaction graph-sequence framework.

## Layout

```text
experiments/
├── config/        YAML configuration files
├── data/
│   └── raw/       TSP and TBM monitoring CSV inputs
├── notebooks/     Exploratory notebooks
├── outputs/       Generated metrics, figures, and checkpoints
├── scripts/       Runnable experiment entry points
├── src/           Reusable Python modules
├── requirements.txt
└── PIPELINE_DIAGRAM.md
```

## Entry Points

Run from this directory:

```powershell
python scripts/mvp1_build_graph.py
python scripts/mvp4_full_model.py
```

`mvp1_build_graph.py` builds representative graph snapshots and visualization
figures.

`mvp4_full_model.py` runs the full training and evaluation pipeline, including
baselines, graph-sequence model, ablations, bootstrap intervals, permutation
tests, prediction plots, and hotspot maps.

## Model Search

Use `scripts/run_model_search.py` to explore accuracy-oriented and
interpretability-oriented configurations around the stratified setup.

Generate trial configs without training:

```powershell
python scripts/run_model_search.py --include-augmented
```

Run a compact search and summarize completed trials:

```powershell
python scripts/run_model_search.py --execute --include-augmented --max-runs 6 --search-epochs 20 --search-patience 4
```

After enough trials finish, produce a full ablation-ready config for the best
completed trial:

```powershell
python scripts/run_model_search.py --final-ablation
```

## Source Modules

```text
src/data/           TSP loading, monitoring preprocessing, geometry alignment
src/graph/          Node, edge, and graph-sequence construction
src/models/         Baselines, GNN encoder, graph-sequence models
src/training/       Datasets, losses, training loops, metrics
src/visualization/  Graph, prediction, and hotspot plotting
```

## Current Reproducibility Note

The old `outputs/mvp4/` run has been removed because it was generated before
several evaluation fixes. The current complete result set is
`outputs/mvp4_stratified/`. To regenerate it, run:

```powershell
python scripts/mvp4_full_model.py --config config/stratified.yaml
```
