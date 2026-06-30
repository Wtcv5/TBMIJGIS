"""Shared IJGIS-style helpers for publication-quality figures.

Canonical implementation synced with .codex/skills/ijgis-scientific-figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ── Color Palette ──

IJGIS_COLORS = {
    "truth": "#1F1F1F",
    "full_model": "#1B6CA8",
    "lstm": "#2A9D8F",
    "tsp_lstm": "#4C9A2A",
    "orange_baseline": "#D95F02",
    "persistence": "#7A7A7A",
    "rock": "#5B7C99",
    "tbm": "#C84C31",
    "edge": "#9A9A9A",
    "accent": "#6A3D9A",
    # TBM component edge colors (consistent with graph_viz.py)
    "cutterhead_edge": "#FF7A1A",
    "shield_edge": "#4169E1",
    # Semantic palettes for manuscript figures (colorblind-safe, grayscale-distinct)
    # Case colours (do not reuse model colours for case identity)
    "case_bsll_h1": "#1B6CA8",
    "case_bsll_h3": "#2A9D8F",
    "case_sjls_h3": "#D95F02",
    # Unified TBM component colours (consistent across all manuscript figures)
    "comp_cutterhead": "#D73027",
    "comp_front_shield": "#FC8D59",
    "comp_middle_shield": "#91BFDB",
    "comp_tail_shield": "#5AAE61",
    # Diagnostic variant colours (raw vs detrended)
    "diagnostic_raw": "#1B6CA8",
    "diagnostic_detrended": "#D95F02",
    # Unified grayscale ramp (replace ad-hoc greys)
    "gray_dark": "#4D4D4D",
    "gray_medium": "#888888",
    "gray_light": "#CFCFCF",
    "gray_lighter": "#E8E8E8",
    "gray_lightest": "#F5F5F5",
    # Schematic fill colours
    "fill_rock": "#EEF3E6",
    "fill_tbm": "#E8F1F6",
    "fill_active": "#F5EFE6",
    "fill_neutral": "#F7F7F7",
}

# Convenience dictionaries for semantic colour access
CASE_PALETTE = {
    "bsll_dyk1017_205": IJGIS_COLORS["case_bsll_h1"],
    "bsll_dyk1017_205_h3": IJGIS_COLORS["case_bsll_h3"],
    "sjls_dyk1252_411": IJGIS_COLORS["case_sjls_h3"],
}

COMPONENT_PALETTE = {
    "cutterhead": IJGIS_COLORS["comp_cutterhead"],
    "front_shield": IJGIS_COLORS["comp_front_shield"],
    "middle_shield": IJGIS_COLORS["comp_middle_shield"],
    "tail_shield": IJGIS_COLORS["comp_tail_shield"],
}

# Preferred colormaps (colorblind-safe)
IJGIS_CMAPS = {
    "sequential": "cividis",      # descriptors and magnitudes
    "sequential_alt": "viridis",  # alternative sequential
    "diverging": "RdBu_r",       # residuals, centered differences
    "rock_geology": "turbo",      # 3D debug renders only (not for manuscript)
}

# Backward-compatible names used by existing visualization modules.
PALETTE = {
    "rock": IJGIS_COLORS["rock"],
    "tbm": IJGIS_COLORS["tbm"],
    "edge": IJGIS_COLORS["edge"],
    "truth": IJGIS_COLORS["truth"],
    "full": IJGIS_COLORS["full_model"],
    "baseline": IJGIS_COLORS["orange_baseline"],
    "accent": IJGIS_COLORS["accent"],
    "lstm": IJGIS_COLORS["lstm"],
    "tsp_lstm": IJGIS_COLORS["tsp_lstm"],
    "persistence": IJGIS_COLORS["persistence"],
}


# ── Figure Sizing ──

def mm_to_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Convert millimeters to inches for Matplotlib figsize."""
    return width_mm / 25.4, height_mm / 25.4


def figure_size(kind: str = "single", aspect: float = 0.68) -> tuple[float, float]:
    """Return IJGIS-friendly figure size in inches.

    kind:
      - "single": compact single-column figure (89 mm)
      - "one_half": wider figure for dense comparisons (130 mm)
      - "double": full-width multi-panel figure (183 mm)
    """
    widths = {
        "single": 89.0,
        "one_half": 130.0,
        "double": 183.0,
    }
    width_mm = widths.get(kind, widths["single"])
    return mm_to_inches(width_mm, width_mm * aspect)


# ── Style Application ──

def apply_ijgis_style() -> None:
    """Apply compact IJGIS-style Matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def apply_publication_style() -> None:
    """Backward-compatible alias for IJGIS-style plotting defaults."""
    apply_ijgis_style()


# ── Panel Labels ──

def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.04) -> None:
    """Add a consistent lowercase bold panel label."""
    label = label.lower()
    kwargs = {
        "fontsize": 8,
        "fontweight": "bold",
        "va": "bottom",
        "ha": "left",
    }
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, label, transform=ax.transAxes, **kwargs)
    else:
        ax.text(x, y, label, transform=ax.transAxes, **kwargs)


# ── TBM Unwrapped Surface Map ──

def setup_unwrapped_surface_axes(
    ax: plt.Axes,
    x_range: tuple[float, float],
    component_boundaries: Optional[list[tuple[float, str]]] = None,
) -> None:
    """Configure axes for a TBM unwrapped surface map.

    Args:
        ax: Matplotlib axes to configure.
        x_range: (x_min, x_max) longitudinal range in meters.
        component_boundaries: list of (x_position, label) pairs for TBM
            component boundary lines. E.g. [(0, "Cutterhead"),
            (3.0, "Front shield"), (6.5, "Middle shield"), (10.0, "Tail shield")].
    """
    ax.set_xlim(*x_range)
    ax.set_ylim(0, 360)
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel(r"Circumferential angle $\theta$ (deg)")

    # Mark geological orientation
    for angle, name in [(0, "Springline"), (90, "Crown"),
                        (180, "Springline"), (270, "Invert")]:
        ax.axhline(angle, color="#D0D0D0", linewidth=0.4, linestyle=":")
        if name in ("Crown", "Invert"):
            ax.text(x_range[1] + 0.3, angle, name, fontsize=6,
                    va="center", ha="left", color="#666666")

    # Draw component boundaries
    if component_boundaries:
        for x_pos, label in component_boundaries:
            ax.axvline(x_pos, color="#888888", linewidth=0.6, linestyle="--")
            ax.text(x_pos + 0.15, 350, label, fontsize=6, va="top",
                    ha="left", color="#444444", rotation=90)


def compute_unwrapped_coords(tbm_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute unwrapped (x, theta) coordinates from 3D TBM surface positions.

    Args:
        tbm_positions: (N, 3) array of TBM surface node positions [x, y, z].

    Returns:
        x: (N,) longitudinal positions.
        theta: (N,) circumferential angles in degrees [0, 360).
    """
    x = tbm_positions[:, 0]
    theta = np.degrees(np.arctan2(tbm_positions[:, 2], tbm_positions[:, 1])) % 360
    return x, theta


# ── Chainage Evolution Figure ──

def setup_chainage_evolution_axes(
    fig: plt.Figure,
    n_monitoring_vars: int = 5,
    kind: str = "one_half",
) -> list[plt.Axes]:
    """Create stacked subplots for chainage evolution figure.

    Args:
        fig: Figure to add subplots to.
        n_monitoring_vars: number of monitoring variable panels.
        kind: figure size kind ("single", "one_half", "double").

    Returns:
        List of axes: [relevance_axis, mon_var_1, ..., mon_var_n].
    """
    n_panels = 1 + n_monitoring_vars
    figsize = figure_size(kind, aspect=0.22 * n_panels)
    gs = fig.add_gridspec(n_panels, 1, hspace=0.15)
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    # Share x-axis across all panels
    for ax in axes[1:]:
        ax.sharex(axes[0])

    # Hide x-tick labels on all but the bottom panel
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    axes[0].set_ylabel(r"Mean $C_j$")
    axes[-1].set_xlabel("Chainage (m)")

    return axes


# ── Multi-Panel Layout Helpers ──

def make_two_panel_prediction(
    kind: str = "one_half", aspect: float = 1.1,
    height_ratio: tuple[float, float] = (2.3, 1.0),
    hspace: float = 0.12,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a 2-panel figure for prediction + residual.

    Returns:
        (fig, ax_main, ax_residual)
    """
    figsize = figure_size(kind, aspect)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratio, hspace=hspace)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)
    return fig, ax_main, ax_res


def make_four_panel(
    kind: str = "double", aspect: float = 0.55,
    wspace: float = 0.22, hspace: float = 0.28,
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """Create a 2x2 panel figure.

    Returns:
        (fig, [[ax_00, ax_01], [ax_10, ax_11]])
    """
    figsize = figure_size(kind, aspect)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=wspace, hspace=hspace)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    return fig, axes


# ── Save ──

def save_publication_figure(fig, path: str | Path | None, dpi: int = 600) -> None:
    """Save a publication figure and close it.

    If the requested path is not PDF, also save a PDF sibling for manuscript use.
    """
    if path is None:
        plt.close(fig)
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    if target.suffix.lower() != ".pdf":
        fig.savefig(target.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def save_figure(fig, save_path: str | Path | None, dpi: int = 600) -> None:
    """Backward-compatible save helper for publication workflows."""
    save_publication_figure(fig, save_path, dpi=dpi)


# ── Audit Helpers ──

def audit_figure_dpi(path: str | Path, min_dpi: int = 300) -> dict:
    """Check if a raster figure meets DPI requirements.

    Returns dict with keys: path, format, dpi, width_px, height_px, passes.
    """
    from PIL import Image

    img = Image.open(path)
    width_px, height_px = img.size
    dpi = img.info.get("dpi", (72, 72))
    dpi_value = min(dpi[0], dpi[1])

    return {
        "path": str(path),
        "format": img.format,
        "dpi": round(dpi_value, 1),
        "width_px": width_px,
        "height_px": height_px,
        "passes": dpi_value >= min_dpi,
    }


def audit_colormap_usage(fig: plt.Figure) -> list[str]:
    """Check if any axes in the figure use non-recommended colormaps.

    Returns list of warnings for axes using jet, rainbow, turbo, etc.
    """
    avoid_cmaps = {"jet", "rainbow", "turbo", "hsv", "gist_rainbow", "nipy_spectral"}
    warnings = []

    for ax in fig.get_axes():
        for collection in ax.collections:
            cmap = getattr(collection, "cmap", None)
            if cmap is not None and cmap.name in avoid_cmaps:
                warnings.append(
                    f"Axis '{ax.get_title()}' uses '{cmap.name}' colormap — "
                    f"prefer 'cividis' or 'viridis' for colorblind safety."
                )
        for image in ax.get_images():
            cmap = getattr(image, "cmap", None)
            if cmap is not None and cmap.name in avoid_cmaps:
                warnings.append(
                    f"Axis '{ax.get_title()}' uses '{cmap.name}' colormap — "
                    f"prefer 'cividis' or 'viridis' for colorblind safety."
                )

    return warnings
