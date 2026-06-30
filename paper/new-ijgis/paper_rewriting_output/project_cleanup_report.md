# Project Cleanup Report

## Active Manuscript

The active revised manuscript is:

- `paper/new-ijgis/revised_tex_project_v2/main_revised.tex`

Supporting active materials:

- `paper/new-ijgis/revised_tex_project_v2/sections_revised/`
- `paper/new-ijgis/revised_tex_project_v2/references.bib`
- `paper/new-ijgis/revised_tex_project_v2/figures/`
- `paper/new-ijgis/revised_tex_project_v2/main_revised.pdf`

## Integrated From Previous Draft

From `paper/ijgis-template/`:

- migrated `references.bib`;
- migrated `reference_doi_audit.md`;
- restored citation anchors into the revised Introduction and Related work;
- copied reusable event-centred result figures into the active manuscript figure folder.

## Deleted As Abandoned Or Generated

- previous manuscript source after migration:
  - `paper/ijgis-template/`
- duplicate outer files:
  - `paper/new-ijgis/main_revised.tex`
  - `paper/new-ijgis/figures_terms_revision_notes.md`
- LaTeX build products from the new manuscript folder:
  - `main_revised.aux`
  - `main_revised.bbl`
  - `main_revised.blg`
  - `main_revised.log`
  - `main_revised.out`
- LaTeX build products from the previous draft folder:
  - `main.aux`
  - `main.bbl`
  - `main.blg`
  - `main.fdb_latexmk`
  - `main.fls`
  - `main.log`
  - `main.out`
- Python cache directories:
  - `experiments/**/__pycache__/`
- local agent traces:
  - `.claude/`
- redundant root-level event file manifest:
  - `tbm_shield_sticking.csv`
- experiment generated outputs:
  - `experiments/outputs/descriptors/`
  - `experiments/outputs/descriptor_diagnostics/`
  - `experiments/outputs/figures/`
- exploratory experiment notebook:
  - `experiments/notebooks/`
- large external SJLS raw text staging directory after processed CSV/audit retention:
  - `experiments/data/sjls_dyk1252_411_raw/`
- duplicate experiment notes merged into `experiments/README.md`:
  - `experiments/EVIDENCE_PROTOCOL.md`
  - `experiments/PIPELINE_DIAGRAM.md`

Note: `.agents/` was removed once but the current execution environment may recreate it as an empty local working directory. It contains no manuscript, experiment, figure, or evidence files and is not treated as part of the cleaned project.

## Intentionally Retained

- `experiments/`: retained as the computational and data-processing base.
  - Configs were reduced to descriptor-pipeline fields.
  - Generated outputs are no longer retained; only `experiments/outputs/README.md` remains.
  - The processed SJLS voxel table and audit are retained; the external raw text files were removed after conversion.
- `event_centered_experiment_figures/`: retained as figure-generation source material.
- `tbm_shield_sticking/`: retained as event-alignment and edge-provenance evidence material, despite historical naming.

## Remaining Cleanup/Revision Items

- Generate final `fig:construction` candidate-relation construction figure.
- Generate final `fig:sensitivity` sensitivity/provenance-overlap figure.
- Confirm the controlling motivation in a dedicated `confirmed_motivation.md` before a deeper structural rewrite.
- Fix wide-table overfull boxes during the final IJGIS formatting pass.
