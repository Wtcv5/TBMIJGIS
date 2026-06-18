| Source | Path/URL | Role | Reliability | Notes |
|---|---|---|---|---|
| User draft skeleton | `ijgis-template/main.tex` | manuscript container | authoritative for file structure | IJGIS-oriented LaTeX skeleton with fallback article class |
| Introduction | `ijgis-template/sections/introduction.tex` | drafted section | substantive, 6 paragraphs | covers engineering context, GIScience elevation, sequence-model limitation, graph-learning rationale, research questions and contributions |
| Methods | `ijgis-template/sections/methods.tex` | drafted section | substantive, 9 subsections | covers framework overview, data alignment, geological/machine inputs, graph construction, constrained edges, edge features, response supervision, interpretation, evaluation design |
| Results | `ijgis-template/sections/results.tex` | placeholder | non-substantive | contains only example table/figure placeholders |
| Discussion | `ijgis-template/sections/discussion.tex` | placeholder | non-substantive | contains only itemized prompts |
| Conclusion | `ijgis-template/sections/conclusion.tex` | placeholder | non-substantive | contains only structural prompts |
| Abstract | `ijgis-template/main.tex` (inline) | revised draft | substantive | revised per abstract_revision.md; still uses prospective evaluation wording |
| Bibliography | `ijgis-template/references.bib` | placeholder | low | sample entries only |
| Manuscript outline | `岩机交互图序列框架_论文提纲与实验方案.md` | primary source for motivation, method design, evaluation plan | authoritative | provides title options, abstract logic, section logic, equations, experiment design |
| Method flow design | `method_chapter_flow_design.md` | primary source for method details | authoritative | provides forecasting formulation, monitoring vectors, GRU design, edge features, geometric prior |
| Publication figures | `experiments/outputs/figures/` | manuscript figure evidence | authoritative | paper-ready PDF figures for framework, graph construction, prediction, ablation, hotspot, evolution, and decision-support views |
| Stratified MVP4 outputs | `experiments/outputs/mvp4_stratified/` | training evidence | authoritative | latest complete stratified run with model weights, metrics, ablations, prediction plots, and hotspot maps |
| Raw monitoring data | `experiments/data/raw/monitoring.csv` | input data | authoritative | TBM monitoring records |
| Raw TSP data | `experiments/data/raw/tsp.csv` | input data | authoritative | TSP geological attributes |
| Metrics module | `experiments/src/training/metrics.py` | evaluation implementation | authoritative | implements MAE, RMSE, R2, Pearson, Spearman |
| IJGIS reference papers | `IJGIS-papaers/` | exemplar corpus | reference only | 9 PDFs for style learning and citation |

## Intake Status

- Target venue: IJGIS (confirmed).
- Motivation: confirmed on 2026-06-15 with GIScience positioning emphasis.
- Manuscript format: LaTeX.
- Sections drafted: Introduction (substantive), Methods (substantive).
- Sections placeholder: Results, Discussion, Conclusion.
- Experimental outputs available: publication PDF figures and the latest stratified MVP4 result set.
- Quantitative metrics: code infrastructure exists (metrics.py); current metric JSON files are in `experiments/outputs/mvp4_stratified/`.
- Bibliography: still placeholder.
- Exemplar learning: completed (exemplar_learning_dossier.md).

## Hard-Gate Assessment

- `source_map.md`: updated.
- `confirmed_motivation.md`: updated and confirmed.
- `target_journal_research.md`: exists, needs minor update.
- `paper_diagnosis.md`: needs update.
- `original_logic_map.md`: needs update.
- `exemplar_learning_dossier.md`: completed.
- `paragraph_function_templates.md`: pending.
- `result_narrative_templates.md`: pending.
- `section_blueprints.md`: needs update for Results/Discussion/Conclusion.
- `style_profile.md`: can be derived from exemplar_learning_dossier.

## Immediate Blockers

1. Specific quantitative metrics from the stratified MVP4 run need to be extracted and organized.
2. Results section cannot be substantively written without organized metric values.
3. Bibliography needs real references aligned with cited literature.
