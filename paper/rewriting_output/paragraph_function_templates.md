# Paragraph Function Templates

Derived from the exemplar learning dossier and calibrated for IJGIS graph-method
papers. Each template specifies the rhetorical job, reader question answered,
and a sentence-level pattern.

---

## Introduction Templates

### I-P1: Domain Problem Setup
- **Rhetorical job**: Establish why the reader should care about the problem.
- **Reader question**: What is the practical context and why does it matter?
- **Pattern**: "[Domain] is [context]. Under [conditions], [phenomenon] may [consequence]. [Observable signals] are [informative but limited], because [specific limitation]."
- **Our version**: "TBM excavation in complex geological settings produces coupled rock--machine responses. Monitoring variables are informative external expressions, but monitoring curves alone provide limited evidence about where interaction occurs, which components are implicated, and how local interaction relations evolve."

### I-P2: GIScience Elevation
- **Rhetorical job**: Elevate the problem from engineering to geographic-information science.
- **Reader question**: Why is this also a GIScience problem?
- **Pattern**: "Recent developments in [sensing/monitoring] have increased data availability. The challenge is therefore not only [engineering concern], but also [representation concern]: heterogeneous sources with different [spatial support/temporal resolution/physical meaning] must be integrated into a representation that preserves [spatial structure] together with [temporal evolution]. At this point, the problem extends beyond [engineering domain] and becomes a [GIScience domain] problem."
- **Our version**: Already drafted; needs strengthening with GIScience citations (spatial representation, heterogeneous data integration, spatial support mismatch).

### I-P3: Representational Gap
- **Rhetorical job**: Define the specific gap in existing methods.
- **Reader question**: Why are current methods insufficient for this paper's goal?
- **Pattern**: "Existing [method family] have shown that [what they can do]. However, most of these methods represent [domain] as [representation type]. [Method family] are effective for [what they do well], but they usually treat [domain] as [simplified representation] rather than as [richer representation]. This limitation matters when the objective is not only [narrow goal], but also [broader goal]."
- **Our version**: Already drafted; well-aligned with the template.

### I-P4: Method-Family Justification with Boundary
- **Rhetorical job**: Justify the chosen method family while setting boundaries.
- **Reader question**: Why this approach, and why not unconstrained?
- **Pattern**: "[Method family] offers a natural alternative because [strength]. [Method family] have shown strong capacity for [capability] in domains such as [examples]. Yet [domain] is not a generic [method family] problem in which [design choice] can be [unconstrained action]. [Constraint] must remain consistent with [domain knowledge]. In this setting, [method family] is most meaningful when it operates on [constrained design] rather than on [unconstrained alternative]."
- **Our version**: Already drafted; well-aligned.

### I-P5: Research Questions and Contributions
- **Rhetorical job**: State the paper's promise precisely.
- **Reader question**: What exactly does this paper contribute?
- **Pattern**: "Motivated by this gap, this study proposes [method description]. [Entity A] are organized as [representation A], while [Entity B] are [representation B]. [Key mechanism] is [described]. Instead of [alternative approach], the framework uses [chosen approach]. The learned [output] is then [projected/mapped] to [interpretable view]. The study addresses [N] questions: [Q1]; [Q2]; [Q3]. The main contributions are [N]fold. First, [C1]. Second, [C2]. Third, [C3]. Fourth, [C4]."
- **Our version**: Already drafted; needs GIScience framing in contribution statements.

---

## Methods Templates

### M-Overview: Framework Overview
- **Rhetorical job**: Orient the reader to the overall workflow.
- **Pattern**: "This section formalizes [framework name] as a [adjective] [representation type] supervised by [supervision signal]. The method represents [domain] as [representation], rather than as [alternative]. Let [notation] be [definition]. The workflow contains [N] stages: [stage list]."
- **Our version**: Already drafted.

### M-Entity: Spatial Entity Definition
- **Rhetorical job**: Define what the spatial entities are and why.
- **Pattern**: "The [entity type A] input is [description] $D_A = \{...\}$, where [notation] denotes [meaning]. The [entity type B] input is represented as [description] $M_B = \{...\}$, where [notation] denotes [meaning]."
- **Our version**: Already drafted.

### M-Graph: Graph Snapshot Definition
- **Rhetorical job**: Formalize the graph structure at each time step.
- **Pattern**: "At each [time step], a graph snapshot is defined as $G_t = (V_t^A \cup V_t^B, E_t^{AA} \cup E_t^{BB} \cup E_t^{AB})$, where [node/edge set definitions]."
- **Our version**: Already drafted.

### M-Constraint: Constrained Edge Construction
- **Rhetorical job**: Explain the geometric screening and why it matters.
- **Pattern**: "A [entity A] and [entity B] are linked at step $t$ only if [condition list]. This design intentionally limits [learning scope]. The [learned quantity] is therefore interpreted as [constrained interpretation], rather than as [overclaim]."
- **Our version**: Already drafted; interpretive boundary statement present.

### M-Supervision: Response-Supervised Learning
- **Rhetorical job**: Explain what is learned and from what signal.
- **Pattern**: "After graph construction, each snapshot is encoded by [encoder] to produce [output]. The supervision signal is provided by [signal description], rather than by [alternative]. Under this design, the model learns [what is learned]. The resulting [output] are therefore [constrained interpretation]."
- **Our version**: Already drafted.

### M-Interpretation: Graph-to-Surface Mapping
- **Rhetorical job**: Explain how latent relevance becomes interpretable.
- **Pattern**: "To recover spatial interpretability, the learned [quantity] is projected back onto [space A] and [space B]. [Space A projection] produces [output A]. [Space B projection] produces [output B]. These outputs show [what they show], while remaining consistent with [supervision signal]."
- **Our version**: Already drafted.

---

## Results Templates

### R-P1: Predictive Comparison
- **Rhetorical job**: Show that the proposed method outperforms baselines.
- **Reader question**: Does the method work for prediction?
- **Pattern**: "Table [N] reports the prediction performance of all models on [dataset]. Compared with [baseline families], the proposed framework achieves [improvement description] in terms of [metrics]. Among the baselines, [best baseline] obtains [values], while the proposed method reaches [values], representing a [percentage] improvement in [metric]."
- **Key constraint**: Must use real numbers; no fabrication.

### R-P2: Structural Ablation
- **Rhetorical job**: Show that each design component matters.
- **Reader question**: Do the graph structure and geometric constraints matter?
- **Pattern**: "Table [N] presents the ablation results. Removing [component A] leads to [degradation], indicating that [component A] contributes to [function]. Replacing [component B] with [alternative] results in [degradation], confirming that [component B's role]. Using a static graph instead of the graph sequence causes [degradation], demonstrating that [temporal evolution matters]."
- **Key constraint**: Must use real numbers; no fabrication.

### R-P3: Spatial Interpretation Output
- **Rhetorical job**: Show what newly becomes visible spatially.
- **Reader question**: What does the spatial interpretation reveal?
- **Pattern**: "Figure [N] shows the projected hotspot maps on the TBM shield surface at [chainage range]. [Observation about spatial pattern]. Figure [N+1] presents the chainage evolution of interaction relevance. [Observation about temporal evolution]. These outputs demonstrate that the framework can [spatial reasoning capability], which is not available from monitoring-only sequence models."
- **Key constraint**: Must use real figures; no fabrication.

---

## Discussion Templates

### D-P1: Scientific Interpretation
- **Rhetorical job**: Explain what the results mean beyond numbers.
- **Pattern**: "The results suggest that the framework captures [what it captures], rather than merely [narrower interpretation]. The spatial interpretation outputs reveal [insight], which aligns with [domain knowledge or GIScience principle]. The ablation results confirm that [key design choice] is essential for [function], supporting the claim that [GIScience contribution]."

### D-P2: Boundary and Caution
- **Rhetorical job**: State what should not be overclaimed.
- **Pattern**: "The learned edge relevance should be interpreted as [what it is], rather than as [what it is not]. [Limitation 1]. [Limitation 2]. These boundaries arise because [reason], and they imply that [caution]."

### D-P3: Broader GIScience Implications
- **Rhetorical job**: Connect to GIScience beyond the specific case.
- **Pattern**: "More broadly, the framework demonstrates how [GIScience principle] can be applied to [domain]. The approach of [method feature] is not limited to TBM excavation but could be adapted to [other domains] where [shared condition]. This contributes to the GIScience literature on [theme] by showing that [contribution]."

---

## Conclusion Template

### C-1: Concise Close
- **Pattern**: "This study proposed [method] for [problem]. The framework [what it does], addressing the gap of [gap]. [Strongest evidence sentence]. [Limitation boundary]. [Future direction or GIScience significance]."
