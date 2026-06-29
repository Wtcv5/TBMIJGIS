# IJGIS Exemplar-Based Revision Report

## Basis

The revision used the local `IJGIS-papaers` folder as the style and quality corpus. The most relevant exemplars were:

- Rahimi et al. (2021), IJGIS: graph data model quality standard.
- Chen et al. (2026), IJGIS: spatially constrained graph-learning quality standard.
- Wang et al. (2025), IJDE: knowledge graph spatiotemporal data model quality standard.
- Hu et al. (2025), IJDE: tunnel digital-twin neighboring-domain standard.

## Expert Diagnosis

1. IJGIS graph papers need a formal spatial representation problem, not only a workflow.
2. The method contribution should be expressed as reusable graph/field operators.
3. Related work should position the paper against spatial graph data models and spatially constrained graph learning, not only generic GNNs or tunnelling prediction.
4. Results should tell readers which method promise each table/figure tests.
5. Claims should remain bounded because the evidence supports diagnostic spatial representation, not calibrated jamming prediction or causal contact inference.

## Changes Made

- Added local IJGIS/IJDE exemplar citations to `references.bib`.
- Strengthened `related_work.tex` with explicit positioning against graph data models, spatiotemporal knowledge graphs, spatially constrained graph learning, and tunnel digital-twin prediction.
- Added a formal method objective in `methods.tex`:
  - graph-construction operator `G`
  - field-computation operator `F`
  - monitoring records as readouts rather than graph-construction inputs.
- Added a Results roadmap in `results.tex` to align result blocks with method promises.
- Preserved the cautious claim boundary: the paper contributes a queryable component-indexed anomalous interaction field, not measured contact force, calibrated jamming probability, or causal attribution.

## Verification

- `bibtex main` completed and generated entries for the new citations.
- `xelatex -interaction=nonstopmode main.tex` completed successfully.
- Final PDF: `paper/ijgis-template/main.pdf`, 29 pages.
- No undefined citations or labels.
- No overfull or underfull boxes detected.
- Remaining warning is the expected XeLaTeX `inputenc` warning.
