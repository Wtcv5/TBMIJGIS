# Evidence Protocol

The current evidence chain is descriptor-based. It does not require trained
prediction checkpoints.

## Formal Descriptor Runs

```powershell
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205.yaml --output-dir outputs/descriptors/bsll_dyk1017_205
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205_h3.yaml --output-dir outputs/descriptors/bsll_dyk1017_205_h3
python scripts/collect_spatial_descriptors.py --config config/sjls_dyk1252_411.yaml --output-dir outputs/descriptors/sjls_dyk1252_411
python scripts/summarize_spatial_descriptors.py
```

## Sensitivity

```powershell
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205.yaml
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_descriptor_sensitivity.py --config config/sjls_dyk1252_411.yaml
```

## Figures

```powershell
python scripts/make_descriptor_figures.py
```

## Evidence Files

- `outputs/descriptors/<case>/component_spatial_descriptors.csv`
- `outputs/descriptors/<case>/descriptor_residual_association.csv`
- `outputs/descriptors/<case>/graph_construction_summary.csv`
- `outputs/descriptors/<case>/sensitivity/descriptor_sensitivity_summary.csv`
- `outputs/descriptors/descriptor_case_summary.csv`
- `outputs/descriptors/descriptor_association_all.csv`

All descriptor values are geometry-weighted TSP anomaly summaries over screened
candidate rock--machine relations. They are diagnostic summaries, not contact
force, contact pressure, jamming probability, or calibrated risk.
