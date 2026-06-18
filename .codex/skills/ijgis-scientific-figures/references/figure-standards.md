# IJGIS-Style Figure Standards

## Purpose

Use these standards to create or revise figures for geospatial and spatial-modeling
papers in the style of the user's provided IJGIS reference papers.

## Design Principles

- Favor analytical clarity over visual decoration.
- Make spatial support explicit: point, voxel, graph node, edge, surface patch,
  chainage interval, grid cell, or trajectory segment.
- Show enough geographic or geometric context for the viewer to understand where
  the modeled relation occurs.
- Keep colors restrained and semantically stable across the paper.
- Use captions and labels to distinguish observation, model prediction, learned
  relevance, and interpretation.

## Sizing and Typography

- Prepare figures for single-column, 1.5-column, or double-column widths.
- Use 7-9 pt text at final size for axis labels, legends, and annotations.
- Use lowercase bold panel labels: `a`, `b`, `c`, ...
- Use consistent variable names with the manuscript, e.g. `AdvanceRate`,
  `Torque`, `Thrust`, `Penetration`, `ShieldPressure`, `chainage`, `G_t`.
- Avoid long labels inside axes. Move detail to captions when labels become dense.

## Color and Encoding

- Use color only for categorical model identity, scalar field intensity, or
  component identity.
- Use a sequential color map for hotspot or magnitude fields.
- Use a diverging color map only for signed residuals or centered differences.
- Keep model colors stable:
  - Full Model: blue (`#1B6CA8`)
  - LSTM/TSP-LSTM: teal/green family (`#2A9D8F` / `#4C9A2A`)
  - XGBoost: orange (`#D95F02`)
  - Persistence: gray (`#7A7A7A`)
  - observed/truth: near black (`#1F1F1F`)
- Avoid rainbow colormaps and excessive saturation.
- Pair color with line style, marker, or direct labels when comparisons matter.

### Colorblind-Friendly Colormaps

- **Preferred sequential**: `cividis` (perceptually uniform, colorblind-safe by
  design). Use for hotspot intensity, attention magnitude, and scalar fields.
- **Alternative sequential**: `viridis` (widely recognized, colorblind-safe).
- **Diverging**: `RdBu_r` (red-blue diverging, readable in grayscale when
  paired with contour lines or hatching).
- **Avoid**: `jet`, `rainbow`, `turbo` (for final manuscript figures — these
  are acceptable only for 3D debug renders where hue range aids depth
  perception, as in PyVista previews).
- **Grayscale test**: before finalizing any color-coded figure, convert to
  grayscale and verify that the key information remains distinguishable.

## Figure-Type Guidance

### Graph Construction

The graph construction figure (typically Fig. 2 in the manuscript) should follow
the IJGIS 4-panel convention used by Scene-GCN, Adaptive Dynamic Graph, and
Region2Vec. Use `plot_graph_construction_figure()` from `graph_viz.py`.

**Panel a — Spatial entity definition (2D longitudinal profile schematic)**:
- Side-view (X-Z plane) showing TBM outline (cutterhead disk + shield cylinder).
- Annotate active zone boundaries: cutterhead look-ahead $L_f$ (orange shading)
  and shield annular zone $\tau$ (blue shading).
- Mark excavated zone (gray shading behind TBM).
- Show rock voxel grid indication ahead of face.
- Label $x_{face}$, $L_f$, $\tau$, $R_s$, and advance direction.
- This is a **schematic diagram**, not a 3D render. Use clean 2D lines and fills.

**Panel b — Edge construction constraints (2D cross-section schematic)**:
- Y-Z cross-section showing shield circle, annular active zone, and candidate edges.
- Illustrate geometry prior filtering: show retained edges (high $\kappa$),
  medium edges, and filtered edges (low $\pi^{rm}$) with different line styles.
- Show surface normal $\mathbf{n}_j$ and its relation to $\kappa$ computation.
- Use legend to explain edge retention criteria.

**Panel c — Graph snapshot structure (adjacency matrix heatmap)**:
- Display the block-structured adjacency matrix: $E_{rr}$ (top-left),
  $E_{mm}$ (bottom-right), $E_{rm}$ (off-diagonal blocks).
- Weight $E_{rm}$ entries by geometry prior $\pi^{rm}_{ij}$; use binary for
  $E_{rr}$ and $E_{mm}$.
- Use a sequential blue colormap (white → light blue → dark blue).
- Label blocks with $V^r$, $V^m$, $E_{rr}$, $E_{mm}$, $E_{rm}$.
- Subsample to max 200 nodes per type for readability.

**Panel d — Graph sequence evolution (multi-chainage adjacency comparison)**:
- Show 3 representative chainages (early, mid, late) side-by-side as small
  adjacency heatmaps.
- Each subplot labeled with chainage value and total edge count.
- Demonstrates how graph structure evolves as TBM advances.

**General rules for graph construction figures**:
- Make rock nodes, TBM nodes, and rock-machine edges visually distinct.
- Downsample dense edges when needed; report full edge counts in caption or text.
- Use transparency for edges so structure is visible without dominating nodes.
- Distinguish cutterhead edges (warm tone, e.g. `#FF7A1A`) from shield edges
  (cool tone, e.g. `#4169E1`) consistently across all panels.
- Prefer 2D schematics and adjacency heatmaps over 3D renders for the main
  manuscript figure. 3D PyVista renders may be used as supplementary material.
- Use `figure_size("double", aspect=0.75)` for the 4-panel layout.

### Prediction and Metrics

- Use chainage on the x-axis for tunnel-process results.
- Align predictions and observations to the same target horizon.
- Plot residuals or confidence intervals only when they are computed correctly.
- Do not overclaim small metric differences without uncertainty or significance.
- Use dashed lines for model predictions and solid lines for observations.
- Include MAE or RMSE in legend entries for quick comparison.

### Ablation

- Order bars or points by experimental logic, not by cherry-picked performance.
  Recommended order: Full Model → No monitoring → No geometry prior →
  Random edges → No geometric constraints → Dynamic graph only.
- Use exact variant names from the methods section.
- Make the removed component explicit: monitoring input, geometry prior,
  randomized interaction edges, or geometric constraints.
- Use horizontal bar charts when variant names are long; vertical bars otherwise.

### Hotspot and Attention Maps

- Label outputs as learned relevance, attention-derived hotspot, or
  response-supervised interaction relevance.
- Do not call hotspots physical contact force, causal source, or ground truth.
- Show TBM component context when projecting to the surface.
- Use the same color scale across comparable hotspot panels unless the caption
  explicitly states per-panel normalization.
- Use `cividis` colormap for all hotspot intensity fields.
- Clip outliers at the 95th percentile for visualization; report the clipping
  threshold in the caption.

### TBM Unwrapped Surface Map

This figure type projects the cylindrical TBM surface onto a 2D rectangle for
spatial analysis of attention/relevance patterns.

- **Coordinate system**:
  - x-axis: longitudinal position along tunnel axis (m), aligned with chainage.
  - y-axis: circumferential angle θ (degrees, 0°–360°), computed as
    `arctan2(z, y)` from the TBM local frame.
- **Component boundaries**: draw horizontal dashed lines at the x-coordinates
  separating cutterhead, front shield, middle shield, and tail shield. Label
  each region.
- **Color encoding**: `cividis` sequential colormap for relevance intensity.
  Add a colorbar labeled "Learned relevance $C_j$" or "Attention-derived
  hotspot intensity".
- **Aspect ratio**: the unwrapped figure should approximate the true
  length-to-circumference ratio of the TBM shield to avoid visual distortion.
  For a 10 m shield with radius 3.95 m, the aspect ratio is approximately
  10 : (2π × 3.95) ≈ 10 : 24.8 ≈ 1 : 2.5.
- **Annotations**: mark the crown (θ = 90°), invert (θ = 270°), and springline
  (θ = 0°, 180°) on the y-axis for geological orientation.
- **Caption template**: "Unwrapped TBM surface map showing response-supervised
  interaction relevance at chainage X.X m. The horizontal axis represents
  longitudinal position along the TBM; the vertical axis represents
  circumferential angle. Dashed lines separate TBM components. Relevance is
  aggregated from learned rock-machine edge attention and indicates
  model-relevant spatial association, not measured contact force."

### Chainage Evolution Figure

This figure tracks how mean hotspot relevance and monitoring responses co-evolve
along the tunnel chainage.

- **Layout**: stacked subplots sharing the x-axis (chainage in meters).
  - Top panel: mean relevance $\bar{C}(x)$ along chainage.
  - Lower panels: aligned monitoring variables (Thrust, Torque, Penetration,
    AdvanceRate, ShieldPressure).
- **x-axis**: chainage (m), shared across all panels.
- **y-axis**: each panel uses its own scale; label with variable name and unit.
- **Color**: use `accent` purple (`#6A3D9A`) for mean relevance; use
  `full_model` blue (`#1B6CA8`) for monitoring traces.
- **Correlation annotation**: optionally report Pearson/Spearman correlation
  between $\bar{C}(x)$ and each monitoring variable in the panel title or as
  an inset.
- **Vertical reference lines**: mark train/val/test split boundaries with
  light gray dashed lines if the figure spans the full dataset.
- **Caption template**: "Co-evolution of mean surface relevance and TBM
  monitoring responses along chainage. Top panel: mean attention-derived
  relevance $\bar{C}$ averaged over all TBM surface nodes at each step.
  Lower panels: observed monitoring variables. Correlation between relevance
  and response does not imply causation."

### Schematics

- Use schematics to explain data flow or representation logic, not software
  architecture details.
- Keep arrows directional and minimal.
- Use mathematical notation sparingly and consistently.
- Prefer vector graphics (PDF/SVG) for schematics; avoid rasterized elements.

## Multi-Panel Layout

### GridSpec Templates

Use `matplotlib.gridspec.GridSpec` for all multi-panel figures. Common layouts:

**2-panel (prediction + residual)**:
```python
fig = plt.figure(figsize=figure_size("one_half", aspect=1.1))
gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.12)
ax_main = fig.add_subplot(gs[0])
ax_res  = fig.add_subplot(gs[1], sharex=ax_main)
```

**4-panel (2×2)**:
```python
fig = plt.figure(figsize=figure_size("double", aspect=0.55))
gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28)
```

**6-panel (graph construction figure)**:
```python
fig = plt.figure(figsize=figure_size("double", aspect=0.65))
gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.0], height_ratios=[1.35, 0.65],
                      wspace=0.12, hspace=0.14)
```

### Panel Label Placement

- Position: `x=-0.08, y=1.04` relative to axes transform (outside top-left).
- Fontsize: 8 pt bold, lowercase letters.
- Use `add_panel_label(ax, "a")` from `style.py` for consistency.

### Spacing Guidelines

- `wspace`: 0.12–0.22 for adjacent columns.
- `hspace`: 0.12–0.28 for adjacent rows.
- Leave more space when subplots have colorbars or legends on the right.
- Use `fig.tight_layout()` as a starting point, then fine-tune with
  `fig.subplots_adjust()`.

## Caption Pattern

Use this structure:

1. What the figure shows.
2. How the data/model output was produced.
3. What comparison or spatial pattern is important.
4. Any limitation needed to avoid overclaiming.

Example:

`Response-supervised interaction hotspots projected onto the TBM surface along
selected chainages. Hotspot intensity is computed from learned rock-machine edge
attention in the final graph snapshot of each input window; it indicates
model-relevant spatial association rather than measured contact force.`

## Output Requirements

- Save line art and plots as PDF.
- Save preview raster images as PNG at 300-600 dpi.
- Use tight bounding boxes without clipping labels.
- Keep generated outputs under `experiments/outputs/<run_name>/`.
- Do not manually edit generated figures unless the edit is documented and
  reproducible.
- Use `pdf.fonttype: 42` and `ps.fonttype: 42` to ensure editable text in
  vector outputs (already set in `apply_ijgis_style()`).
