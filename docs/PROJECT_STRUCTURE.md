# Project Structure Notes

This repository is organized around two deliverables:

1. a reproducible experiment pipeline for rock-TBM interaction graph sequences;
2. an IJGIS-style manuscript that frames the method as a GIScience contribution.

## Recommended Boundaries

Keep these responsibilities separate:

```text
experiments/
  config/       Parameter files for reproducible runs
  data/         Raw and processed experiment data
  notebooks/    Exploratory analysis only
  outputs/      Generated figures, metrics, and model checkpoints
  scripts/      Runnable pipeline entry points
  src/          Reusable experiment package code

paper/
  ijgis-template/      LaTeX manuscript source
  rewriting_output/    Writing diagnostics and revision artifacts
  PaperSpine/          Writing-assistant skill/tool material
  *.md                 Planning and manuscript-development notes

IJGIS-papaers/
  *.pdf         Related papers and target-journal exemplars
```

## Naming Conventions

- Use `snake_case.py` for Python modules and scripts.
- Use clear MVP script names only for runnable milestones, e.g.
  `mvp1_build_graph.py` and `mvp4_full_model.py`.
- Keep generated files under `experiments/outputs/<run_name>/`.
- Keep raw input files under `experiments/data/raw/`.
- Keep manuscript-only data or notes under `paper/` only when they are not used by
  experiment scripts.

## Cleanup Guidance

The following items are safe to ignore in versioned or formal archives:

- `__pycache__/`
- `.venv/`
- LaTeX temporary files such as `.aux`, `.log`, `.out`
- large regenerated outputs under `experiments/outputs/`
- model checkpoints such as `.pt`

Avoid moving `experiments/data/raw/`, `experiments/config/`, or `experiments/src/`
without updating the script path assumptions in `experiments/scripts/`.

