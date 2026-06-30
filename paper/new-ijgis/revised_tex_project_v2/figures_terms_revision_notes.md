# Figures, terminology and IJGIS-style revision notes (v2)

## 1. Structural self-review

The revised manuscript is closer to an IJGIS-style spatial data modelling paper if it is framed as **representation validation** rather than a TBM prediction paper. The strongest IJGIS contribution is the explicit spatial relation layer between TSP-derived rock voxels and component-labelled TBM surface nodes.

### Related work

The previous version had four short subsections. This is acceptable in principle, but it reads fragmented. The v2 revision collapses Related work into three subsections:

1. **TBM jamming and monitoring-based response modelling**
2. **Geological prediction and the component-relation gap**
3. **Spatial relation graphs for moving excavation scenes**

This is more consistent with IJGIS articles, where Related work usually supports a small number of conceptual gaps rather than many narrow literature bins.

### Workflow position

The workflow should not be Figure 1. Figure 1 should introduce the research scene/object. The workflow is now moved into Methodology after the entity table, so readers first see what entities and supports exist, and then see how they flow through graph construction, index computation and provenance queries.

## 2. Figure plan

### Figure 1: Research scene and object of representation (Introduction)
Purpose: show the scene, not the method workflow.

Required elements:
- longitudinal tunnel axis and excavation direction;
- advancing TBM with cutterhead, front shield, middle shield and tail shield labels;
- TSP-derived rock voxels / low-velocity anomaly zone around the tunnel;
- candidate rock--TBM relation edges from voxels to components;
- monitoring records shown as aligned responses, not as graph nodes.

Current file: `figures/fig_research_scene.pdf`.

### Figure 2: Method workflow (Methodology)
Purpose: data -> graph -> component index -> provenance.

Recommended panels:
A. TSP voxel field and TBM surface nodes in a common excavation coordinate.  
B. Candidate-edge filtering: active zone, distance, normal compatibility, excavation state.  
C. Component-level geological anomaly index.  
D. Edge-level provenance query.

### Figure 3: Geometry-constrained candidate relation construction
Purpose: make the edge-screening rules visually auditable.

### Figure 4: Event alignment
Purpose: event chainage, TSP anomaly, F37 fault influence zone, local cross-section and time-chainage alignment.

### Figure 5: Dynamic graph snapshots
Purpose: show pre-event, near-jamming and jammed-state relation support.

### Figure 6: Component-index heatmap and monitoring strips
Purpose: align component indices and TBM responses in the same excavation coordinate.

### Figure 7: Sensitivity and provenance robustness
Purpose: show robustness across parameter settings and top-k edge overlap.

### Figure 8: Edge-level provenance
Purpose: demonstrate that the index is traceable to rock voxels, TBM surface nodes and candidate edges.

## 3. Terminology system

| Chinese concept | Recommended English term | Use | Avoid |
|---|---|---|---|
| 卡机 | TBM jamming / TBM shield jamming | Use **TBM shield jamming** for the studied event; use **TBM jamming** for the general accident category | shield sticking as main term |
| 卡住状态 | jammed state | State descriptor | sticking state |
| 临近卡机窗口 | near-jamming window | Event-development interval | near-stuck window |
| 岩体体素 | rock voxel | TSP-derived geological spatial support | block if not block-model context |
| TBM表面节点 | TBM surface node | Component-labelled machine-side spatial support | sensor node unless it is a physical sensor |
| 候选岩体-TBM关系 | candidate rock--TBM relation / candidate edge | Screened spatial relation | contact edge, force edge |
| 几何兼容权重 | geometry-compatibility weight | Edge relation weight | contact force, friction weight |
| 地质异常得分 | geological anomaly score | Voxel attribute q_i | risk score unless calibrated |
| 组件级地质异常指数 | component-level geological anomaly index | Main index I_c(t) | fault probability |
| 几何暴露度 | geometric exposure | A_c(t) | physical exposure without definition |
| 边级溯源 | edge-level provenance | Traceability of interpretation | explainability alone |

## 4. Jamming terminology note

Literature usage supports **TBM jamming** and **shield jamming** as the more standard terms. `Sticking` appears mainly in expressions such as cutterhead sticking, while shield-side obstruction is typically described as shield jamming. Therefore the manuscript now uses **TBM shield jamming** for the event and reserves `jammed state` for the observed state.

Suggested bibliographic anchors to add to the final reference file:

- Ramoni and Anagnostou, *Tunnel boring machines under squeezing conditions* / *Thrust force requirements for TBMs in squeezing ground*.
- Zhang and Zhou, *Time-dependent jamming mechanism for Single-Shield TBM tunneling in squeezing rock*, Tunnelling and Underground Space Technology, 2017.
- Huang et al., *Mechanism and forecasting model for shield jamming during TBM tunnelling through deep soft ground*, European Journal of Environmental and Civil Engineering, 2019.
- Hou et al., *Prediction of shield jamming risk for double-shield TBM tunnels based on numerical samples and random forest classifier*, Acta Geotechnica, 2023.
- Wang et al., *A jamming risk warning model for TBM tunnelling based on Bayesian statistical methods*, Scientific Reports, 2025.

## 5. Main text changes in v2

- Added an Introduction figure for research object/scene.
- Changed the main event term from `shield sticking` to `TBM shield jamming`.
- Collapsed Related work from four subsections into three.
- Moved workflow from first figure position to Methodology after the entity table.
- Preserved the representation-validation logic: relation reconstruction, component specificity, alternative summaries, sensitivity and edge-level provenance.
