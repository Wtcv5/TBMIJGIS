## Revised Abstract

TBM excavation is commonly modeled as a temporal monitoring sequence, but such
representations offer limited spatial explicitness regarding where rock--machine
interactions occur, which TBM components are implicated, and how these interactions
evolve along chainage. This study proposes a response-supervised, geometry-constrained
graph sequence framework for spatially explicit representation and interpretation of
rock--TBM interaction. TSP-derived geological attributes are converted into rock voxel
nodes, while the cutterhead and shield are represented as parameterized TBM surface
nodes. Candidate rock--machine edges are screened using spatial proximity,
signed-normal consistency, active-zone filtering, and excavation-state constraints, so
that learning is restricted to geometrically plausible interaction relations. Rather
than relying on sparse jamming labels or manually calibrated contact forces, the
framework uses continuously recorded TBM monitoring responses to supervise the relative
importance of geometry-screened rock--machine edges. A graph encoder extracts
interaction-aware representations from each excavation snapshot, and a temporal encoder
captures the evolution of graph states along excavation. Learned edge relevance is then
projected back onto TBM surface and chainage spaces to produce response-consistent
shield hotspot maps and interaction-evolution views. The evaluation is structured
against monitoring-only sequence models, TSP-augmented sequence models, tabular
baselines, and structural graph ablations, including randomized rock--machine edges
and removal of geometric constraints. By unifying geological sensing, machine
geometry, monitoring response, and spatial interpretation within a graph sequence, the
framework establishes a spatially explicit basis for analyzing rock--TBM interaction
dynamics and supporting interpretable excavation-risk assessment.

## What Changed

- Moved the opening toward a GIScience-style spatial-representation problem.
- Compressed graph-construction details into a tighter causal sequence.
- Reframed the closing around spatial explicitness and interpretation rather than a generic computational basis.
- Preserved prospective evaluation wording to avoid fabricating results that are not yet supplied.
