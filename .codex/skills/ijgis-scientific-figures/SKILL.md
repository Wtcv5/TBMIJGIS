---
name: ijgis-scientific-figures
description: Create, audit, and polish IJGIS-style scientific figures for geospatial, spatial graph, spatiotemporal model, and TBM interaction manuscripts. Use when Codex needs to improve Matplotlib/Python figures, graph visualizations, prediction plots, ablation plots, hotspot maps, unwrapped surface maps, chainage evolution figures, manuscript figure panels, captions, output formats, or publication-quality visual consistency based on the user's provided IJGIS reference papers and the current project's experiment outputs.
---

# IJGIS Scientific Figures

## Core Workflow

Use this skill when working on publication figures for this project or similar
IJGIS-style geospatial manuscripts.

1. Inspect the target figure's role in the paper: method schematic, data map,
   graph construction, model result, ablation, prediction trace, spatial hotspot,
   unwrapped surface map, chainage evolution, or uncertainty/significance plot.
2. Read `references/figure-standards.md` before changing visual design, captions,
   or plotting defaults.
3. Prefer editing reusable plotting code in `experiments/src/visualization/`
   rather than manually altering generated PDFs or PNGs.
4. Preserve vector outputs for line art and plots. Save PDF for manuscript use
   and PNG at high DPI for quick review.
5. Verify that the figure supports the paper's GIScience claim: spatial entity
   representation, relation plausibility, graph-sequence evolution, or
   interpretable geospatial learning.

## Project-Specific Figure Types

- **Graph construction figures**: show rock voxels, TBM surface nodes, and
  geometry-screened interaction edges without visual clutter.
- **Prediction figures**: compare Full Model, LSTM/TSP-LSTM, XGBoost, and
  Persistence on aligned chainage axes; avoid implying causality.
- **Ablation figures**: emphasize structural contribution and geometry constraints,
  not only marginal accuracy differences.
- **Hotspot figures**: present attention-derived relevance as response-supervised
  interpretation, not ground-truth contact force.
- **Unwrapped surface maps**: project cylindrical TBM surface onto 2D (x vs θ)
  with component boundaries and geological orientation markers (crown, invert,
  springline). Use `cividis` colormap.
- **Chainage evolution figures**: stacked subplots showing co-evolution of mean
  relevance and monitoring responses along chainage, with correlation annotations.
- **Pipeline/schematic figures**: reduce algorithm steps to readable panels with
  consistent notation and panel labels.

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

- The figure answers one paper question and has no decorative elements.
- Axes, legends, colorbars, and panel labels remain readable at final column size.
- Units and variables match manuscript notation.
- Color encodes meaning consistently across figures.
- The palette is colorblind-aware (`cividis` for sequential, `RdBu_r` for diverging)
  and printable in grayscale where possible.
- Line plots and diagrams are exported as PDF; raster outputs are at least 300 dpi.
- Multi-panel labels use lowercase `a`, `b`, `c`, ... and are positioned consistently.
- Captions state what is plotted, how it was produced, and what conclusion is safe.
- Hotspot/attention plots avoid causal or force-estimation language unless validated.
- Unwrapped surface maps include component boundaries and geological orientation.
- Chainage evolution figures share x-axis and label each panel's variable.
- No axes use `jet`, `rainbow`, or `turbo` colormaps (use `audit_colormap_usage()`).

## Local IJGIS References

Use the PDFs in `IJGIS-papaers/` as local style exemplars. Do not overfit to a
single paper. Extract common conventions: restrained colors, compact panels,
clear geospatial context, explicit model comparison, and concise captions.
