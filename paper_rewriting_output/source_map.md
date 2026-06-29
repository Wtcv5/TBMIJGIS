# Source Map

| Source | Path/URL | Role | Reliability | Notes |
|---|---|---|---|---|
| User manuscript | `paper/ijgis-template/*.tex` | Claims, methods, results, limitations | Authoritative for this paper | No new experimental claims should be added without script/table support. |
| Experiment scripts and outputs | `experiments/scripts`, `experiments/outputs` | Source of computed diagnostics and figures | Authoritative for this paper | Used only for results already generated in this project. |
| IJGIS exemplar PDFs | `IJGIS-papaers/*.pdf` | Venue-quality style and argument structure | Style/positioning only | Used to learn section logic, claim strength, and graph-method presentation. |
| Extracted exemplar text | `paper_rewriting_output/ijgis_text/*.txt` | Searchable text for exemplar learning | Derived from local PDFs | PDF extraction warnings did not prevent text output. |
| Bibliography | `paper/ijgis-template/references.bib` | Citation keys and metadata | Must match cited sources | New local exemplar citations are added only when metadata is visible in PDFs. |

Target venue: International Journal of Geographical Information Science.

Confirmed motivation: the paper contributes a chainage-referenced spatial graph representation and component-indexed anomalous interaction field for TBM excavation, not a high-accuracy predictor or calibrated jamming-risk model.

User priorities: sharpen the paper as a method/algorithm contribution, avoid seven-step workflow framing, avoid treating monitoring responses as graph nodes, and align the revision with IJGIS-quality graph/spatiotemporal data-model papers.
