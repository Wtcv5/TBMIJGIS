# Descriptor Pipeline

```text
TSP voxels + TBM monitoring + TBM surface geometry
        |
        v
Chainage-referenced spatial entity model
        |
        v
Geometry-constrained rock-machine candidate relations
  - active zone
  - distance threshold and decay
  - surface-normal compatibility
  - excavation state
        |
        v
Component-chainage descriptors
  A_c(t): geometric exposure
  I_c(t): geometry-weighted TSP anomaly intensity
        |
        v
Persistence residual diagnostics
  e_{t+h}^{(k)} = r_{t+h}^{(k)} - r_t^{(k)}
        |
        v
Descriptor-residual association, threshold sensitivity, publication figures
```

Main scripts:

- `scripts/collect_spatial_descriptors.py`
- `scripts/summarize_spatial_descriptors.py`
- `scripts/run_descriptor_sensitivity.py`
- `scripts/make_descriptor_figures.py`
