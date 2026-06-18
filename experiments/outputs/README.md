# Experiment Outputs

This directory contains generated figures, metrics, and model checkpoints.

## Current Contents

- `mvp4_stratified/`: latest complete stratified experiment run.
- `figures/`: publication-oriented PDF figures referenced by the manuscript.

Old quick runs, stale `mvp4/` outputs, scratch graph-construction figures, logs,
and temporary files have been removed.

Regenerate the current full experiment from `experiments/` with:

```powershell
python scripts/mvp4_full_model.py --config config/stratified.yaml
```
