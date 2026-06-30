# Generated Experiment Outputs

This directory is the target for regenerated experiment outputs. In the cleaned project, generated descriptor tables, diagnostics, and figures are not retained here.

Recreate formal outputs from `experiments/` with:

```powershell
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205.yaml --output-dir outputs/descriptors/bsll_dyk1017_205
python scripts/collect_spatial_descriptors.py --config config/bsll_dyk1017_205_h3.yaml --output-dir outputs/descriptors/bsll_dyk1017_205_h3
python scripts/collect_spatial_descriptors.py --config config/sjls_dyk1252_411.yaml --output-dir outputs/descriptors/sjls_dyk1252_411
python scripts/summarize_spatial_descriptors.py
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205.yaml
python scripts/run_descriptor_sensitivity.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_descriptor_sensitivity.py --config config/sjls_dyk1252_411.yaml
python scripts/make_descriptor_figures.py
```
