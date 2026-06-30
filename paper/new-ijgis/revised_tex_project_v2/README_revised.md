# Revised TeX project v2

This version incorporates a self-review against IJGIS-style structure and the terminology correction from `shield sticking` to `TBM shield jamming`.

Main changes:

1. Added an Introduction scene figure (`figures/fig_research_scene.pdf`) to explain the research object before the method workflow.
2. Moved the method workflow to the Methodology section after the entity model table.
3. Collapsed Related work into three subsections to reduce fragmentation.
4. Replaced the main event term with `TBM shield jamming`; `jammed state` is used only as a state descriptor.
5. Updated figure and terminology notes in `figures_terms_revision_notes.md`.

Active manuscript entry point:

- `main_revised.tex`
- section files in `sections_revised/`
- bibliography in `references.bib`
- manuscript figures in `figures/`

Compile with:

```bash
pdflatex main_revised.tex
bibtex main_revised
pdflatex main_revised.tex
pdflatex main_revised.tex
```

The current figure outputs combine the new research-scene figure with reusable event-centred figures from the previous IJGIS draft. The remaining schematic placeholders are `fig:construction` and `fig:sensitivity`; generate final versions after the graph-construction and sensitivity panels are fixed.
