---
name: ijgis-scientific-figures
description: Create, audit, and polish IJGIS-style scientific figures for geospatial, spatial graph, spatiotemporal model, and TBM interaction manuscripts. Use when Codex needs to improve Matplotlib/Python figures, graph visualizations, prediction plots, ablation plots, hotspot maps, unwrapped surface maps, chainage evolution figures, manuscript figure panels, captions, output formats, or publication-quality visual consistency based on the user's provided IJGIS reference papers and the current project's experiment outputs.
---

# IJGIS Scientific Figures

## Core Workflow

Use this skill when working on publication figures for this project or similar
IJGIS-style geospatial manuscripts.

1. Inspect the manuscript logic before choosing figures. Identify the exact
   claim each figure must support and the claim it must avoid.
2. Inspect the target figure's role in the paper: method schematic, data map,
   graph construction, model result, ablation, prediction trace, spatial hotspot,
   unwrapped surface map, chainage evolution, or uncertainty/significance plot.
3. Read `references/figure-standards.md` before changing visual design, captions,
   or plotting defaults.
4. Prefer editing reusable plotting code in `experiments/src/visualization/`
   rather than manually altering generated PDFs or PNGs.
5. Preserve vector outputs for line art and plots. Save PDF for manuscript use
   and PNG at high DPI for quick review.
6. Verify that the figure supports the paper's GIScience claim: spatial entity
   representation, relation plausibility, graph-sequence evolution, or
   interpretable geospatial learning.

## Flexible Figure Portfolio

Do not hard-code a fixed number of manuscript figures. Rich figure sets are
acceptable when each figure earns its place in the argument. Decide the figure
portfolio from evidence roles:

- Keep a figure in the main text when it is needed to understand the method,
  compare the cases, or evaluate a central claim.
- Move a figure to supplementary material when it repeats the same evidence
  role, shows per-variable detail, or documents robustness without changing the
  narrative.
- Reject or redesign a figure when it is visually polished but cannot be linked
  to a defensible claim in the manuscript.
- Prefer a compact multi-panel figure over several isolated figures only when
  the panels form one reasoning step. Do not merge unrelated evidence just to
  reduce figure count.

## Manuscript Logic Gate

Before creating or revising any figure, map it to the current paper logic:

- The paper is a GIScience representation and interpretation paper, not a
  broadly superior forecasting-model paper.
- BSLL DyK1017+205 supports compact-case prediction checks, multi-step behavior,
  and surface/chainage interpretation. It must not be used to claim a robust
  topology-driven accuracy gain.
- SJLS Dyk1252+411 supports the clearer geometry-sensitive prediction effect and
  external TSP case evidence. Its accuracy gain is small, so figures must show
  scale, uncertainty, and ablation context.
- Learned hotspots are response-supervised relevance patterns. They are not
  measured contact force, causal geological source, operational risk, or alert
  labels unless independent validation exists.
- The strongest visual evidence should connect spatial entities, constrained
  relations, graph evolution, prediction/ablation checks, and interpretable
  surface patterns.

Every candidate figure should pass this short test:

```text
Figure role:
Manuscript claim supported:
Case(s): BSLL / SJLS / both
Evidence type: representation / prediction / ablation / spatial statistic / interpretation
Boundary sentence needed in caption:
Main text or supplement:
```

## Journal-Learned Style Rules

Recent IJGIS, IJDE, and Geo-spatial Information Science articles continue to
publish spatial graph learning, deep spatiotemporal forecasting, Earth-system
deep learning, and geospatial graph autoencoder work. The transferable figure
pattern is not minimal figure count; it is controlled analytical density:

- Lead with spatial context and representation, not only model architecture.
- Use multi-panel figures to show a progression: data support -> graph/relation
  construction -> model output -> spatial interpretation.
- Pair spatial maps/surfaces with quantitative summaries when the claim depends
  on both geography and metrics.
- Keep model-comparison plots restrained, aligned on a meaningful spatial or
  temporal axis, and explicit about small effect sizes.
- Use captions to state the safe interpretation and the limitation, especially
  for attention, relevance, risk, and case-study generalization.

## Project-Specific Figure Types

- **Graph construction figures**: show rock voxels, TBM surface nodes, and
  geometry-screened interaction edges without visual clutter.
- **Case context figures**: show BSLL and SJLS as two real tunnel scenarios,
  including start chainage, sample intervals, TSP support, and monitoring
  coverage. Do not label them new/old data.
- **Prediction figures**: compare Full Model, LSTM/TSP-LSTM, XGBoost, and
  Persistence on aligned chainage axes; avoid implying causality.
- **Ablation figures**: emphasize structural contribution and geometry constraints,
  not only marginal accuracy differences.
- **Spatial evidence figures**: summarize Moran's I, degree-control correlation,
  relevance-geology association, and relevance-response association across
  cases. These are often more important than raw forecast traces.
- **Hotspot figures**: present attention-derived relevance as response-supervised
  interpretation, not ground-truth contact force.
- **Unwrapped surface maps**: project cylindrical TBM surface onto 2D (x vs θ)
  with component boundaries and geological orientation markers (crown, invert,
  springline). Use `cividis` colormap.
- **Chainage evolution figures**: stacked subplots showing co-evolution of mean
  relevance and monitoring responses along chainage, with correlation annotations.
- **Pipeline/schematic figures**: reduce algorithm steps to readable panels with
  consistent notation and panel labels.
- **Scenario/decision figures**: allowed only as exploratory model-diagnostic
  views. Avoid "risk", "alert", "warning", or "decision support" wording unless
  there are independent operational labels and calibration evidence.

## Rich Figure Portfolio Patterns

Use these as combinable patterns, not as a fixed list:

- **Method-first ordering**: start the main figure sequence with the transferable
  GIScience framework, then show spatial entity formalisation, geometry
  constraints, and graph-sequence network architecture. Introduce BSLL/SJLS only
  after the common method is established.
- **Representation spine**: spatial entities, relation constraints, graph
  sequence, and tensor/model inputs.
- **Two-case context**: BSLL and SJLS tunnel intervals, TSP/monitoring support,
  train/test chainage coverage, and target horizon.
- **Prediction and ablation**: compact case-level comparison that makes the
  small SJLS advantage and mixed BSLL evidence visible without exaggeration.
- **Spatial statistics**: Moran's I, geometry-only baseline, degree correlation,
  relevance-geology correlation, and response association in one readable panel
  family.
- **Surface interpretation**: selected unwrapped surface maps or hotspot panels
  with component boundaries and consistent color scales.
- **Chainage evolution**: relevance and observed monitoring variables aligned
  along chainage, preferably for BSLL multi-step/explanation evidence and an
  SJLS contrast panel when useful.
- **Supplementary completeness**: per-variable traces, all chainages, additional
  horizons, seeds, and extra hotspot snapshots.

## Matplotlib Helper

The canonical plotting style lives in the project's own module:

```text
experiments/src/visualization/style.py
```

This Skill's `scripts/ijgis_mpl_style.py` is the reference implementation that
`style.py` is synced with. All new functions added to the Skill should also be
copied to `style.py`.

Key functions available from `style.py`:

| Function | Purpose |
|---|---|
| `apply_ijgis_style()` / `apply_publication_style()` | Set rcParams for IJGIS |
| `figure_size(kind, aspect)` | Get figsize in inches (single/one_half/double) |
| `add_panel_label(ax, label)` | Add lowercase bold panel label |
| `setup_unwrapped_surface_axes(ax, ...)` | Configure TBM unwrapped surface map axes |
| `compute_unwrapped_coords(positions)` | Convert 3D → (x, θ) for unwrapped map |
| `setup_chainage_evolution_axes(fig, n)` | Create stacked chainage evolution subplots |
| `make_two_panel_prediction(...)` | Create prediction + residual figure |
| `make_four_panel(...)` | Create 2×2 panel figure |
| `save_publication_figure(fig, path)` | Save PDF + PNG, close figure |
| `audit_colormap_usage(fig)` | Check for non-colorblind-safe colormaps |
| `audit_figure_dpi(path)` | Check raster figure DPI |

Color constants: `IJGIS_COLORS` (model/component colors), `IJGIS_CMAPS`
(recommended colormaps), `PALETTE` (backward-compatible aliases).

## Audit Checklist

Before considering a figure manuscript-ready, check:

- The figure answers a manuscript question and has no decorative elements.
- The figure is not kept or removed because of a fixed figure count; its role in
  the evidence chain is explicit.
- Axes, legends, colorbars, and panel labels remain readable at final column size.
- Units and variables match manuscript notation.
- Color encodes meaning consistently across figures.
- The palette is colorblind-aware (`cividis` for sequential, `RdBu_r` for diverging)
  and printable in grayscale where possible.
- Line plots and diagrams are exported as PDF; raster outputs are at least 300 dpi.
- Multi-panel labels use lowercase `a`, `b`, `c`, ... and are positioned consistently.
- Captions state what is plotted, how it was produced, and what conclusion is safe.
- Hotspot/attention plots avoid causal or force-estimation language unless validated.
- Risk, alert, and decision-support language is absent unless independently
  validated.
- Unwrapped surface maps include component boundaries and geological orientation.
- Chainage evolution figures share x-axis and label each panel's variable.
- Preview PNGs and grayscale previews have been visually inspected; automated
  audits are not enough.
- No axes use `jet`, `rainbow`, or `turbo` colormaps (use `audit_colormap_usage()`).

## Local IJGIS References

Use the PDFs in `IJGIS-papaers/` as local style exemplars. Do not overfit to a
single paper. Extract common conventions: restrained colors, compact panels,
clear geospatial context, explicit model comparison, and concise captions.
