# Interpretation Evidence Bank

This file records the current quantitative evidence for the paper's
interpretability-oriented argument. Use these claims conservatively: attention
hotspots indicate response-supervised model relevance, not measured contact
force or causal ground truth.

## Evidence Sources

- `experiments/outputs/evidence/interpretation_summary.csv`
- `experiments/outputs/evidence/bsll_dyk1017_205/posthoc_evidence.json`
- `experiments/outputs/evidence/bsll_dyk1017_205_h3/posthoc_evidence.json`
- `experiments/outputs/evidence/sjls_dyk1252_411/posthoc_evidence.json`
- `experiments/outputs/evidence/*/component_relevance.csv`

## Supported Claims

| Claim | Evidence | Allowed Strength | Notes |
|---|---|---|---|
| The graph-sequence model produces spatially structured surface relevance rather than spatially random hotspot fields. | Mean Moran's I is positive in all three evaluated cases: BSLL h=1 `0.546`, BSLL h=3 `0.302`, SJLS `0.300`. | Moderate | State as learned spatial organization, not physical contact validation. |
| Learned relevance is not simply a node-degree artifact. | Mean Pearson correlation between node relevance and incident edge count is negative: BSLL h=1 `-0.229`, BSLL h=3 `-0.324`, SJLS `-0.456`. | Moderate | This supports degree-normalized relevance, but does not prove causality. |
| In SJLS, learned relevance is strongly aligned with the TSP-derived velocity field along the test chainage. | Relevance-Geology Pearson `0.870`, Spearman `0.752`, with small p-values in `outputs/evidence/sjls_dyk1252_411/posthoc_evidence_summary.csv`. | Strong for this case | Phrase as association between model relevance and TSP velocity context. |
| Geometry-only relevance is less response-adaptive than the learned relevance. | SJLS geometry-only Moran's I `0.196` vs learned Moran's I `0.300`; geometry-only TSP correlation returned `0.0` under the current profile computation. | Moderate | Useful as support for response-supervised learning beyond fixed geometry. |
| Dominant relevant TBM components differ by case/horizon. | Dominant component: BSLL h=1 middle shield share `0.288`; BSLL h=3 middle shield share `0.291`; SJLS tail shield share `0.275`. | Descriptive | Component shares are close; avoid overinterpreting dominance. |
| Relevance-response association differs across cases. | Strongest response correlation: BSLL h=1 AdvanceRate abs Pearson `0.924`; BSLL h=3 AdvanceRate `0.665`; SJLS Torque `0.938`. | Descriptive to moderate | This can structure case comparison, but test sample counts are small. |

## Claims To Avoid

- Do not claim the hotspot is a measured contact-force map.
- Do not claim causal attribution from attention alone.
- Do not claim globally significant prediction improvement across both tunnels.
- Do not claim BSLL proves geometry improves prediction accuracy; BSLL evidence is mainly interpretive.

## Results Narrative Implication

The Results section should move from prediction consistency to spatial
interpretation:

1. Prediction metrics establish that the response-supervised model remains
   competitive with strong temporal baselines, with SJLS showing a small
   geometry-sensitive gain.
2. Post-hoc relevance analysis shows that learned surface hotspots are spatially
   organized and not reducible to candidate-edge degree.
3. Case comparison shows that the same graph-sequence representation can expose
   different relevance-response associations across tunnel settings.
