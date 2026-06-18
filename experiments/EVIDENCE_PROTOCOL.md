# Evidence Protocol

This file defines the reproducible evidence chain used to support model
accuracy, spatial interpretation, and robustness claims.

## 1. Current Formal Run

The current formal result directory is:

```text
outputs/mvp4_stratified/
```

It was generated from:

```powershell
python scripts/mvp4_full_model.py --config config/stratified.yaml
```

Key files:

- `metrics_global.json`
- `metrics_per_variable.json`
- `ablation_metrics.json`
- `bootstrap_ci.json`
- `significance_tests.json`
- `best_graph_model.pt`

## 2. Post-Hoc Spatial Evidence

Regenerate checkpoint-based spatial evidence without retraining:

```powershell
python scripts/collect_evidence.py --config config/stratified.yaml --run-dir outputs/mvp4_stratified --output-dir outputs/evidence
```

This writes:

- `outputs/evidence/posthoc_evidence.json`
- `outputs/evidence/posthoc_evidence_summary.csv`

The script rebuilds the test set, loads `best_graph_model.pt`, recomputes
predictions, extracts edge relevance, and computes:

- attention-geology correlation;
- Moran's I of TBM surface relevance;
- component-level coefficient of variation;
- relevance-vs-degree control correlation;
- a geometry-only baseline from fixed candidate-edge priors.

## 3. Stability Runs

Use model search to generate and run multi-seed or parameter-sensitivity trials:

```powershell
python scripts/run_model_search.py --execute --include-augmented --max-runs 6 --search-epochs 80 --search-patience 15
```

Completed trials are summarized in:

```text
outputs/search/summary.csv
outputs/search/summary.json
```

After enough trials finish, write a full ablation config for the best completed
trial:

```powershell
python scripts/run_model_search.py --final-ablation
```

## 4. Manuscript Rule

Only values with a direct JSON/CSV source in `outputs/mvp4_stratified/`,
`outputs/evidence/`, or `outputs/search/` should be reported as experimental
evidence. Exploratory or synthetic runs must be labelled as such in the paper.
