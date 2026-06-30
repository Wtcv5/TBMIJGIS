# LaTeX Report

## Compile Target

- Working directory: `paper/new-ijgis/revised_tex_project_v2/`
- Entry point: `main_revised.tex`
- Engine sequence:
  - `xelatex -interaction=nonstopmode main_revised.tex`
  - `bibtex main_revised`
  - `xelatex -interaction=nonstopmode main_revised.tex`
  - `xelatex -interaction=nonstopmode main_revised.tex`

## Result

- Output PDF: `paper/new-ijgis/revised_tex_project_v2/main_revised.pdf`
- PDF length after integration: 21 pages
- Bibliography resolved: yes
- Cross-references resolved: yes
- Figure files found: yes

## Remaining Warnings

- `inputenc` is ignored by XeLaTeX; harmless under the fallback article class.
- Overfull boxes remain in several wide tables:
  - `sections_revised/methods.tex`, lines 40--54
  - `sections_revised/study_area_event_centred.tex`, lines 37--48 and 77--86
  - `sections_revised/results_event_centred.tex`, lines 35--46
- These are layout issues, not missing-source or citation failures. They should be fixed during the final IJGIS formatting pass by narrowing table text, using smaller fonts, or moving some wide tables to landscape/supplementary material.

## Notes

- BibTeX produced the `.bbl` successfully. MiKTeX also attempted to write a user-level BibTeX log and reported a permission warning; this did not block bibliography generation.
- The remaining figure placeholders in the manuscript are the candidate-relation construction figure and sensitivity figure. They are intentionally retained until final figure assets are generated.
