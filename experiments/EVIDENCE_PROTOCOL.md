# Evidence Protocol

This protocol defines which generated results can be used in the manuscript and
how to regenerate them.

## Formal Runs

Run from `experiments/`:

```powershell
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205.yaml
python scripts/run_graph_sequence_case.py --config config/bsll_dyk1017_205_h3.yaml
python scripts/run_graph_sequence_case.py --config config/sjls_dyk1252_411.yaml
```

Formal run directories:

- `outputs/bsll_dyk1017_205/`
- `outputs/bsll_dyk1017_205_h3/`
- `outputs/sjls_dyk1252_411/`

Each formal run should contain:

- `metrics_global.json`
- `metrics_per_variable.json`
- `ablation_metrics.json`
- `bootstrap_ci.json`
- `significance_tests.json`
- `preprocessing_audit.json`
- `best_graph_model.pt`

## Post-Hoc Spatial Evidence

Regenerate checkpoint-derived interpretation evidence without retraining:

```powershell
python scripts/collect_evidence.py --config config/bsll_dyk1017_205.yaml --run-dir outputs/bsll_dyk1017_205 --output-dir outputs/evidence/bsll_dyk1017_205
python scripts/collect_evidence.py --config config/bsll_dyk1017_205_h3.yaml --run-dir outputs/bsll_dyk1017_205_h3 --output-dir outputs/evidence/bsll_dyk1017_205_h3
python scripts/collect_evidence.py --config config/sjls_dyk1252_411.yaml --run-dir outputs/sjls_dyk1252_411 --output-dir outputs/evidence/sjls_dyk1252_411
python scripts/summarize_interpretation_evidence.py
```

The evidence collector rebuilds the test set, loads `best_graph_model.pt`,
extracts attention-derived TBM surface relevance, and writes:

- `posthoc_evidence.json`
- `posthoc_evidence_summary.csv`
- `component_relevance.csv`

## Manuscript Rule

Only values traceable to JSON/CSV files under the formal output directories or
`outputs/evidence/` should be reported as experimental evidence. Exploratory
runs must be labelled as exploratory.
