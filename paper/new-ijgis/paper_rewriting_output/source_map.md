# Source Map for `paper/new-ijgis`

| Source | Path/URL | Role | Reliability | Notes |
|---|---|---|---|---|
| Active revised manuscript | `paper/new-ijgis/revised_tex_project_v2/main_revised.tex` | Current manuscript entry point | Authoritative current draft | Uses `sections_revised/`, `figures/`, and `references.bib`. |
| Revised section files | `paper/new-ijgis/revised_tex_project_v2/sections_revised/` | Current prose and LaTeX section source | Authoritative current draft | Keeps the v2 terminology shift from shield sticking to TBM shield jamming. |
| Figure and terminology notes | `paper/new-ijgis/revised_tex_project_v2/figures_terms_revision_notes.md` | Figure plan and terminology control | Authoritative design notes | Records final figure intentions and terms to avoid. |
| Previous IJGIS draft | `paper/ijgis-template/` | Historical draft and source of reusable citations/figures | Removed after migration | Used to migrate `references.bib`, DOI audit, citation anchors, and reusable event-centred figures, then deleted from the cleaned project. |
| Bibliography | `paper/new-ijgis/revised_tex_project_v2/references.bib` | Current bibliography for new manuscript | Current working bib | Copied from the previous IJGIS draft after DOI audit. |
| DOI audit | `paper/new-ijgis/revised_tex_project_v2/reference_doi_audit.md` | Bibliography provenance note | Supporting audit | Copied from the previous IJGIS draft. |
| Event-centred generated figures | `event_centered_experiment_figures/` | Source scripts, tables, and generated figure assets | Supporting evidence/figure production | Reusable figures were copied into the active manuscript figure folder. |
| Experiment code and data | `experiments/` | Data processing, graph construction, diagnostics, and figure-generation code | Authoritative computational source | Keep as the reproducibility base for manuscript claims. |
| TBM shield-jamming validation package | `tbm_shield_sticking/` | Event-alignment data, provenance tables, and plotting scripts | Supporting event evidence | Naming is historical; manuscript term is TBM shield jamming. |
| Project terminology note | `terminology_system_dynamic_rock_tbm_spatial_relation_graph.md` | Concept and terminology control | Supporting note | Useful for future language consistency. |

## Current Motivation Status

The working motivation visible in the manuscript is:

> represent the moving spatial relations between TSP-derived geological voxels and component-labelled TBM surfaces so that a TBM shield-jamming event can be interpreted at component level with edge-level provenance.

This has not yet been separately confirmed by the author in `confirmed_motivation.md`; do that before a full structural rewrite.

## Cleanup Policy

- Keep `paper/new-ijgis/revised_tex_project_v2/` as the active manuscript.
- Do not keep `paper/ijgis-template/`; the useful material has been migrated into `paper/new-ijgis/revised_tex_project_v2/`.
- Remove LaTeX build products (`*.aux`, `*.log`, `*.out`, `*.bbl`, `*.blg`, `*.fls`, `*.fdb_latexmk`) from manuscript folders.
- Remove duplicate outer files under `paper/new-ijgis/` once their content exists inside `revised_tex_project_v2/`.
- Do not remove experiment or event-evidence folders unless their outputs are demonstrably regenerated elsewhere.
