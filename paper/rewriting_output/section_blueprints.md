# Section Blueprints After 2026-06-24 Repositioning

The manuscript spine is now: geometry-constrained rock--TBM spatial interaction
graphs + component--chainage descriptors + persistence-residual consistency.
GNN--GRU is an exploratory readout only.

| Section | Motivation Job | Target Move Sequence | Evidence Inputs | Required Changes |
|---|---|---|---|---|
| Title/Abstract | State that the paper is about spatial interaction graphs and diagnostic residual consistency, not deep prediction. | Monitoring models preserve time but miss spatial relations; define graph entities and candidate relations; derive descriptors; evaluate against persistence residuals; bound claims. | User-provided repositioning note; existing case names. | Avoid GNN-GRU, learned relevance, prediction improvement as central terms. |
| 1 Introduction | Make graph representation necessary. | TBM responses are spatially coupled; monitoring curves are temporally useful but spatially silent; rock voxels and TBM components form a heterogeneous spatial relation problem; contributions. | Existing intro, user note. | Replace response-supervised learning spine with explicit descriptor spine. |
| 2 Related work | Support the representation gap. | TBM response analysis; TSP geological representation; graph-based spatial relation modelling. | Current related-work section and bibliography. | Reduce emphasis on deep spatiotemporal prediction SOTA; emphasise graph data models and relation modelling. |
| 3 Methodology | Define the explicit model. | 3.1 chainage-referenced spatial entities; 3.2 geometry-constrained candidate relation graph; 3.3 component--chainage descriptors $A_c(t)$ and $I_c(t)$; 3.4 response-residual consistency. | Existing graph construction code; user-provided formulas. | Promote $\pi_{ij}^{rm}$ to $w_{ij}^{rm}(t)$; define $q_i=-z(Vp_i)$; remove GNN-GRU from the main experimental route. |
| 4 Case studies and experimental design | Explain data and evaluation without promising prediction superiority. | Case roles; TSP voxelisation; TBM surface sampling; graph construction settings; diagnostic metrics. | Current study-area section; config files; outputs. | Add descriptor-generation settings after code is implemented. |
| 5 Results | Test the new promises. | 5.1 constructed graph sequences; 5.2 component--chainage interaction patterns; 5.3 association with persistence residuals; 5.4 sensitivity to graph-construction thresholds. | Descriptor outputs under `experiments/outputs/descriptors`. | Use $A_c(t)$, $I_c(t)$ maps and residual association tables as main evidence. |
| 6 Discussion | Resolve why this is not a prediction paper. | Why graph representation is necessary; what descriptors add beyond monitoring curves; interpretation boundary and future validation. | Descriptor outputs and persistence-residual logic. | Avoid contact-force, jamming-risk, causal, and prediction-superiority claims. |
| 7 Conclusion | Close on representation and diagnostic workflow. | Restate graph model; restate descriptors and residual consistency; name missing descriptor-level evidence as next step. | Revised Methods/Results. | Do not conclude with prediction improvement. |

## Result Evidence Still Needed

| Needed Output | Purpose | Suggested Source |
|---|---|---|
| Graph construction summary by case and chainage | Show graph is actually constructed, not conceptual. | Cached graph snapshots or rerun graph construction. |
| Component--chainage maps of $A_c(t)$ and $I_c(t)$ | Main diagnostic result. | New descriptor script using $w_{ij}^{rm}(t)$ and $q_i=-z(Vp_i)$. |
| Descriptor--residual association table | Main response-consistency evidence. | Persistence residuals from monitoring targets. |
| Threshold/anomaly-score sensitivity | Robustness. | Vary $\tau_{\mathrm{edge}}$, $\eta_{\min}$, and $q_i$ definitions. |
