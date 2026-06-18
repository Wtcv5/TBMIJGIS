# Exemplar Learning Dossier

## Purpose

This dossier extracts reusable writing strategies from IJGIS and related journal
papers that are methodologically or topically adjacent to the current manuscript.
Each entry identifies a rhetorical move, its structural role, and how it can be
adapted for the rock--TBM interaction graph-sequence paper.

---

## Exemplar 1: Scene-GCN (Li et al. 2025, IJGIS 39(10): 2211--2235)

**Full title**: Scene-GCN: a time-series prediction method in complex monitoring
environments through spatial--temporal knowledge graph (ST-KG)

**DOI**: 10.1080/13658816.2025.2479183

### Why this exemplar matters

- Same journal (IJGIS), same method family (GCN + temporal encoder), same
  application domain (complex monitoring environment prediction).
- Uses a knowledge-graph representation to organize heterogeneous spatial and
  temporal entities before learning, which parallels our graph-sequence
  construction.
- Introduces a "process--state" coupling model for monitoring networks, directly
  comparable to our "rock--machine interaction edge" design.

### Extracted rhetorical moves

| Move | Where | Pattern | Adaptation for our paper |
|---|---|---|---|
| Problem elevation | Abstract opening | "Current forecasting methods for complex geographic scenes often fail to adequately account for both intrinsic factors and external environmental variables" | Open with: "Current TBM response modeling methods often fail to adequately account for spatial interaction structure between rock and machine" |
| Representation-first framing | Introduction | Frames the contribution as a *representation* problem (ST-KG design) before the *learning* problem (GCN architecture) | Our Introduction should foreground the spatial-representation gap before introducing graph learning |
| Knowledge-structure formalization | Methods 3.1--3.2 | Defines entity types (geo-process, geographic state) and their relations formally before any neural architecture | We should formalize rock voxels, TBM surface nodes, and candidate edges as spatial entities with explicit relation rules |
| Dual validation | Results | Validates both prediction accuracy and knowledge-graph structural utility | We should show both predictive metrics and spatial interpretability evidence |
| Boundary statement | Discussion | Explicitly states what the knowledge graph does and does not represent physically | We must state that learned edge relevance is response-consistent, not contact-force |

### Sentence-level patterns to borrow

1. "This study proposes [model name], a [method family]-based model that utilizes [representation innovation] to integrate [input A] and [input B] with [spatial--temporal context] to predict [target]."
2. "First, [entity type] are abstracted as [representation unit], and [observed values] are represented as [state unit], forming a coupled [model name] model tailored for [application domain]."
3. "Environmental knowledge, both dynamic and static, is then embedded within the nodes of [representation], capturing [temporal influence] through [mechanism]."

---

## Exemplar 2: Adaptive Dynamic Graph Learning (Zhu et al. 2025, IJGIS)

**Full title**: Adaptive dynamic graph learning for forecasting urban multimodal flow

**DOI**: 10.1080/13658816.2025.2595655

### Why this exemplar matters

- Directly addresses dynamic graph structure learning, which is our core
  methodological contribution (graph snapshots evolving along chainage).
- Introduces time-varying multimodal graph learning, comparable to our
  excavation-step-dependent graph snapshots.
- Published in IJGIS, so the framing and depth are calibrated to the same
  audience.

### Extracted rhetorical moves

| Move | Where | Pattern | Adaptation for our paper |
|---|---|---|---|
| Static-graph limitation | Introduction | "Current forecasting models, which typically rely on static or manually defined graph structures, are inadequate for capturing dynamic spatial heterogeneity" | "Current TBM models, which typically rely on fixed monitoring sequences, are inadequate for capturing spatially evolving rock--machine interaction" |
| Three-innovation structure | Abstract | Lists exactly three key innovations with (i), (ii), (iii) | Our abstract can list: (i) geometry-constrained edge construction, (ii) response-supervised interaction learning, (iii) graph-to-surface interpretation |
| Module-by-module method | Methods | Each innovation gets its own subsection with formal notation | Our Methods already follows this; keep the pattern |
| Ablation by component | Results | Removes each innovation to show its marginal contribution | Our ablation design (randomized edges, removed constraints, static graph) mirrors this |

### Sentence-level patterns to borrow

1. "To address these limitations, this study introduces [model name], a novel deep learning model that is designed for [task]."
2. "[Model name] introduces [N] key innovations: (i) [module 1] based on [principle], (ii) [module 2] that [function], and (iii) [module 3] for [purpose]."
3. "Experimental results demonstrate that [model] significantly outperforms [baseline families] in terms of [metrics], confirming the effectiveness of [key design choice]."

---

## Exemplar 3: USGT -- Unsupervised Spatial Graph Transformer (IJGIS)

**Full title**: USGT: an unsupervised spatial graph transformer for detecting spatial
communities in human mobility data

### Why this exemplar matters

- Combines spatial constraints with graph learning in an unsupervised setting.
- Our framework also constrains graph learning (geometry screening) and uses
  indirect supervision (monitoring response rather than interaction labels).
- The "spatial community" concept parallels our "interaction hotspot" concept.

### Extracted rhetorical moves

| Move | Where | Pattern | Adaptation for our paper |
|---|---|---|---|
| Unsupervised justification | Introduction | Explains why labeled data is unavailable and why unsupervised/indirect supervision is necessary | We should justify why direct interaction labels are unavailable and why response supervision is the practical alternative |
| Spatial constraint integration | Methods | Embeds spatial constraints directly into the graph learning objective rather than post-hoc | Our geometric prior and edge screening serve the same role; emphasize this integration |
| Interpretation as output | Results | Treats community detection results as interpretable spatial outputs, not just accuracy metrics | Our hotspot maps and chainage evolution views should be presented as primary interpretive outputs, not supplementary |

---

## Exemplar 4: Prediction of Human Activity Intensity via GCN (Li & Gao 2021, IJGIS 35(12))

**Full title**: Prediction of human activity intensity using the interactions in
physical and social spaces through graph convolutional networks

**DOI**: 10.1080/13658816.2021.1912347

### Why this exemplar matters

- Classic IJGIS paper combining GCN (spatial) + LSTM (temporal) for prediction.
- Integrates multiple interaction types (physical + social) in a fused graph,
  comparable to our heterogeneous rock--machine graph.
- 49 CrossRef citations -- a well-received model for IJGIS graph papers.

### Extracted rhetorical moves

| Move | Where | Pattern | Adaptation for our paper |
|---|---|---|---|
| Multi-interaction fusion | Introduction | "Physical interactions and social interactions between spatial units were integrated into a fused graph convolutional network" | "Geological attributes and machine geometry are integrated into a geometry-constrained heterogeneous graph" |
| Spatial + temporal decoupling | Methods | Separate GCN for spatial and LSTM for temporal, then combine | Our GNN encoder + GRU temporal encoder follows the same decoupled design |
| Large-scale validation | Results | Uses country-scale dataset to demonstrate generality | We should emphasize the full-tunnel excavation sequence as a spatially extensive validation |

### Sentence-level patterns to borrow

1. "In this method, [interaction type A] and [interaction type B] between [spatial units] were integrated into a [model type] to model [pattern type] spatial interaction patterns."
2. "The future [target] variation was predicted by combining the [spatial pattern] and the [temporal pattern] of [target] series."
3. "The results demonstrated that our proposed deep learning method with combining [spatial module] and [temporal module] outperformed other baseline approaches."

---

## Exemplar 5: Data-Knowledge Hybrid Driven Tunnel Prediction (Hu et al. 2025, IJDE)

**Full title**: Data-knowledge hybrid driven intelligent prediction method of tunnel
excavation profiles geometric deformation

**DOI**: 10.1080/17538947.2025.2459317

### Why this exemplar matters

- Same application domain (tunnel excavation), same publisher family
  (Taylor & Francis), same hybrid data-knowledge philosophy.
- Introduces knowledge extraction and knowledge base construction for tunnel
  excavation, which is the closest existing work to our knowledge-structured
  graph representation.
- Published in IJDE (sister journal to IJGIS), so the framing is relevant.

### Extracted rhetorical moves

| Move | Where | Pattern | Adaptation for our paper |
|---|---|---|---|
| Data-knowledge hybrid | Introduction | "It is difficult to directly deduce the interactions and relationships between these numerous factors from the data, resulting in challenges related to interpretability" | Use this to justify why pure data-driven models miss spatial interaction structure |
| Knowledge base construction | Methods | Constructs a domain knowledge base before model training | Our graph construction (geometric constraints, active zone) serves as the knowledge-structure layer |
| Interpretability claim | Discussion | Emphasizes that knowledge integration improves both accuracy and interpretability | We should make the same dual claim: geometry constraints improve both prediction and spatial interpretability |

---

## Cross-Exemplar Synthesis: IJGIS Writing Style Profile

### Common structural patterns across IJGIS graph-method papers

1. **Abstract**: 150--250 words. Opens with a limitation of current methods
   (1--2 sentences), states the proposed method and its key innovation (2--3
   sentences), summarizes evaluation (1--2 sentences), closes with significance
   (1 sentence). Uses numbered innovations (i), (ii), (iii) when there are
   multiple contributions.

2. **Introduction**: 5--6 paragraphs following a consistent arc:
   - P1: Domain problem and practical motivation
   - P2: Data/representation challenge (elevates to GIScience)
   - P3: Limitation of existing methods (specific, not strawman)
   - P4: Why the proposed method family is appropriate (with boundary)
   - P5: Research questions and contributions (numbered)

3. **Methods**: Organized by module/subsection. Each module:
   - States what it does and why
   - Provides formal notation
   - Connects to the overall framework
   - Notes interpretive boundaries where relevant

4. **Results**: Three-part structure:
   - Prediction/comparison against baselines (tables + metrics)
   - Structural ablation (what each component contributes)
   - Interpretation output (spatial maps, visualizations)

5. **Discussion**: Three-part structure:
   - Scientific interpretation (what the results mean beyond numbers)
   - Boundary and caution (what should not be overclaimed)
   - Broader GIScience implications (why this matters beyond one case)

6. **Conclusion**: Concise (1 paragraph or 4--5 numbered points):
   - Restate problem and gap
   - State direct answer
   - Name strongest evidence
   - Note limitation boundary

### Tone and style norms

- Formal but not stiff. Uses first-person plural ("we propose") sparingly;
  passive voice is acceptable for method description.
- Avoids superlatives ("first", "novel" used only when genuinely defensible).
- Equations are numbered and referenced in text.
- Figures are referenced in order and each serves a specific argumentative role.
- Citations are used to support claims, not as decoration.
- Interpretability claims are hedged: "response-consistent" rather than
  "physically validated", "spatial relevance" rather than "true interaction".

### Key differences from engineering-journal style

- IJGIS papers foreground the *representation* and *spatial reasoning*
  contribution, not the engineering application.
- The Introduction must explain why the problem is a GIScience problem, not
  just an engineering problem.
- Methods must formalize spatial entities, relations, and constraints
  explicitly, not just describe a neural architecture.
- Results must include spatial interpretability evidence, not just prediction
  metrics.
- Discussion must connect to broader GIScience themes: spatial representation,
  relation plausibility, interpretable geospatial learning.

---

## Actionable Takeaways for Our Paper

1. **Abstract**: Restructure to open with the spatial-representation limitation,
   list three innovations with (i)(ii)(iii), close with GIScience significance.
2. **Introduction**: Already follows the 5-paragraph arc; tighten P2 to make the
   GIScience elevation more explicit with citations from these exemplars.
3. **Methods**: Already well-structured; add explicit interpretive boundary
   statements after each module (following Scene-GCN pattern).
4. **Results**: Adopt the three-part structure (prediction, ablation,
   interpretation) from the cross-exemplar synthesis.
5. **Discussion**: Adopt the three-part structure (interpretation, boundary,
   broader implications) and connect to spatial representation and interpretable
   geospatial learning themes.
6. **Conclusion**: Keep concise; end on the GIScience contribution, not the
   engineering application.
