# Result Narrative Templates

Templates for writing Results and Discussion sections when quantitative evidence
becomes available. These templates enforce the PaperSpine rule: no fabrication,
no placeholder numbers.

---

## Results Narrative Structure

### R1: Prediction Comparison Table Narrative

**Required inputs**: Table with MAE, RMSE, R2 for each model and each target
variable.

**Template**:
```
Table [N] summarizes the prediction performance of all models across [N_target]
response variables. The proposed geometry-constrained graph sequence framework
achieves the best overall performance, with MAE of [X.XXX], RMSE of [X.XXX],
and R2 of [X.XXX] averaged across all targets. Among the monitoring-only
sequence baselines, [best_seq_model] obtains the closest performance with
MAE of [X.XXX] and R2 of [X.XXX], but still falls short of the proposed
method by [X%] in MAE and [X%] in R2. TSP-augmented sequence models improve
over monitoring-only baselines, confirming that geological information
contributes to response prediction, yet their performance remains below the
graph-sequence framework because [reason]. Tabular baselines show the weakest
performance, indicating that [insight].
```

**Per-variable narrative** (if space permits):
```
For [specific variable, e.g., Thrust], the proposed method achieves the
largest improvement over baselines ([X%] in MAE), suggesting that [variable-
specific insight]. For [another variable], the improvement is smaller but
still consistent, indicating that [insight].
```

### R2: Ablation Study Narrative

**Required inputs**: Table with ablation variants and their metrics.

**Template**:
```
Table [N] reports the structural ablation results. Removing geometric
constraints (distance threshold, normal compatibility, active-zone filtering)
from the candidate edge construction causes MAE to increase from [X.XXX] to
[X.XXX] and R2 to decrease from [X.XXX] to [X.XXX], confirming that
geometry-based screening restricts learning to plausible rock--machine
relations and improves both prediction and interpretability. Randomizing
rock--machine edges while preserving the total edge count leads to
[degradation description], indicating that the structured candidate set
carries meaningful spatial information that random connectivity cannot
replace. Replacing the graph sequence with a static graph (using only the
current snapshot without temporal encoding) results in [degradation],
demonstrating that the evolution of graph states along chainage contributes
to response prediction. Removing the geometric prior from edge features
causes [degradation], suggesting that [insight].
```

### R3: Spatial Interpretation Narrative

**Required inputs**: Hotspot maps on TBM surface, chainage evolution plots.

**Template**:
```
Figure [N] presents the response-consistent hotspot maps projected onto the
TBM shield surface at selected chainage positions. [Observation: e.g., "High
interaction relevance concentrates on the front shield and lower quadrants
at chainage X+XXX m, consistent with the known geological transition zone"].
Figure [N+1] shows the chainage evolution of interaction relevance across
shield components. [Observation: e.g., "The interaction emphasis migrates
from the cutterhead to the front shield as excavation advances through the
transition zone, and the hotspot intensity peaks at chainage X+XXX m before
decreasing"]. These spatial patterns are not recoverable from monitoring-only
sequence models, which produce only chainage-indexed scalar predictions
without component-level or circumferential resolution. The hotspot maps
therefore demonstrate that the graph-sequence representation supports spatial
reasoning about interaction structure, fulfilling the interpretability
objective stated in the Introduction.
```

---

## Discussion Narrative Structure

### D1: Scientific Interpretation Narrative

**Template**:
```
The prediction and ablation results together suggest that the geometry-
constrained graph sequence captures spatially structured interaction
information that monitoring-only temporal models cannot access. The
improvement over TSP-augmented sequence models indicates that simply adding
geological features to a sequence input is insufficient; the geological
attributes must be organized into spatially explicit entity--relation
structures that preserve local rock--machine proximity and geometric
plausibility. The spatial interpretation outputs further reveal that the
learned edge relevance is not uniformly distributed but concentrates on
spatially meaningful regions, supporting the claim that the framework
performs interpretable relation learning rather than black-box correlation.
```

### D2: Boundary and Caution Narrative

**Template**:
```
Several boundaries should be noted. First, the learned edge relevance
represents response-consistent interaction importance under geometric
screening, not direct physical contact force or stress. The framework
identifies which candidate relations are most relevant for explaining
response variation, but it does not estimate contact pressure or jamming
probability. Second, the geometric constraints ensure plausibility within
the defined thresholds, but different threshold values may alter the
candidate set and the resulting attention distribution. Third, the
evaluation is conducted on a single tunnel project; cross-project validation
is needed before claiming generalizability. Fourth, the current TBM surface
parameterization uses a simplified cylindrical model; more detailed geometry
may change the node distribution and edge structure.
```

### D3: Broader GIScience Implications Narrative

**Template**:
```
More broadly, this framework demonstrates how spatially explicit graph
representations can be constructed for domains where heterogeneous spatial
entities must be organized into a plausible, interpretable structure under
incomplete observability. The principle of geometry-constrained candidate
relation screening is not limited to TBM excavation; it applies to any
setting where spatial entities interact under geometric compatibility
constraints (e.g., building--environment interaction, infrastructure--terrain
coupling, sensor--phenomenon proximity). The use of continuous monitoring
response as indirect supervision, rather than sparse event labels, offers a
practical path to interpretable relation learning in domains where direct
interaction labels are unavailable. This contributes to the GIScience
literature on spatial representation, relation plausibility, and
interpretable geospatial learning by showing that constrained graph
sequences can support both prediction and spatial reasoning.
```

---

## Usage Rules

1. **No fabrication**: Replace [X.XXX] placeholders only with real numbers from
   actual experimental runs.
2. **No speculation in Results**: The Results section reports what was observed;
   interpretation belongs in Discussion.
3. **Hedge appropriately**: Use "suggests", "indicates", "is consistent with"
   rather than "proves", "confirms", "demonstrates" for inferential claims.
4. **GIScience framing**: Every Results and Discussion paragraph should connect
   back to the spatial representation and interpretable learning contribution,
   not just prediction accuracy.
