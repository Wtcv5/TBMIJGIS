# Rock-TBM Interaction Graph Sequence Project

This project develops a spatially explicit graph-sequence framework for
representing and interpreting rock-TBM interactions during tunnel excavation.

The core idea is to model TSP-derived rock voxels and parameterized TBM surface
nodes as heterogeneous spatial entities, connect them with geometry-constrained
candidate interaction edges, and use TBM monitoring responses to supervise
interaction relevance. The project contains both experiment code and manuscript
materials for an IJGIS-oriented paper.

## Directory Layout

```text
.
├── experiments/          # Reproducible experiment pipeline and model code
├── paper/                # Manuscript source, writing notes, and revision outputs
├── IJGIS-papaers/        # Target-journal and related-literature PDFs
├── andrej-karpathy-skills/ # External writing/coding guidance material
├── CLAUDE.md             # General agent behavior notes
└── README.md             # This project overview
```

## Main Experiment Entry Points

Run commands from `experiments/` unless otherwise noted.

```powershell
python scripts/mvp1_build_graph.py
python scripts/mvp4_full_model.py
```

`mvp1_build_graph.py` builds graph snapshots and graph-construction figures.

`mvp4_full_model.py` runs the full pipeline: graph sequence construction,
baselines, graph-sequence model training, ablations, metrics, attention extraction,
and publication-style figures.

## Current Status Note

Old intermediate and stale experiment outputs have been removed. The currently
kept generated results are:

- `experiments/outputs/mvp4_stratified/`: latest complete stratified experiment
  run with metrics, checkpoints, ablations, and diagnostic plots;
- `experiments/outputs/figures/`: publication-oriented PDF figures used by the
  manuscript.

For a fresh full run, execute `python scripts/mvp4_full_model.py --config
config/stratified.yaml` from `experiments/`.

## Documentation

- [Experiment README](experiments/README.md)
- [Paper README](paper/README.md)
- [Pipeline diagram](experiments/PIPELINE_DIAGRAM.md)
- [Project structure notes](docs/PROJECT_STRUCTURE.md)
