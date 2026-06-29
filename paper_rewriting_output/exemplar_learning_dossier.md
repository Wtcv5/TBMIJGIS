# IJGIS Exemplar Learning Dossier

## Selected Exemplar Roles

| Exemplar | Role Learned | Transfer To This Manuscript |
|---|---|---|
| Rahimi et al. 2021, topology-based graph data model | A spatial graph paper can be accepted as a data-model contribution when objects, relations, queryability, and implementation evidence are explicit. | Present rock voxels, TBM nodes, candidate edges, component aggregation, and edge query as a coherent spatial data model. |
| Chen et al. 2026, USGT | Strong IJGIS graph-method papers define the spatial graph problem, state why generic graph methods are insufficient, then evaluate with baselines/ablations and an applied case. | Add a clearer problem formulation and make results test graph construction, field computation, response alignment, and ablation boundaries. |
| Wang et al. 2025, knowledge graph spatiotemporal data model | Application-oriented graph papers justify the model through heterogeneous data, semantic/spatiotemporal associations, and scenario-based expression. | Position monitoring records as aligned records/readouts, not graph entities; emphasise heterogeneous support alignment. |
| Hu et al. 2025, tunnel digital-twin prediction | Tunnel digital-twin papers often make strong prediction claims; the present paper should contrast itself by narrowing the claim to spatial representation and diagnostic readout. | Keep predictive and risk-calibration claims bounded; use tunnel digital-twin literature only as neighboring context. |

## Learned Quality Standard

1. Abstract should have problem/gap, method mechanism, evidence, and bounded significance; it should not enumerate a long operational workflow.
2. Introduction should move from phenomenon to representational gap to design response; contributions should be algorithmic or data-model units.
3. Methods should include a formal problem/objective statement before implementation details.
4. Results should be organised around method promises, not around whatever tables happen to exist.
5. Discussion should distinguish representation evidence from prediction, causality, or operational decision support.

## Manuscript-Specific Revision Targets

| Weakness Found | IJGIS-Informed Fix |
|---|---|
| Related work still underuses local IJGIS graph data-model precedents. | Add targeted citations to spatial graph data models and spatially constrained graph learning. |
| Methods state the single line but do not yet formalise the computational objective strongly enough. | Add a compact objective/operator definition for graph construction and field computation. |
| Results are technically rich but need a stronger opening that tells readers which method promise each result tests. | Add a result roadmap tied to constructed graph support, field structure, residual alignment, robustness, and traceability. |
| Some statistical terms can sound too confirmatory for the sample size. | Keep permutation and residual analyses as diagnostics; avoid overclaiming diagnostic gain. |
