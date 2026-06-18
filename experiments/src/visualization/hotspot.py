"""TBM surface hotspot visualizations for response-supervised interpretation."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.visualization.style import (
    IJGIS_CMAPS,
    apply_publication_style,
    compute_unwrapped_coords,
    save_figure,
    setup_unwrapped_surface_axes,
)


def aggregate_attention_to_surface(attention: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   n_tbm_nodes: int,
                                   epsilon: float = 1e-8) -> tuple:
    """Aggregate rock-machine edge attention to TBM surface nodes.

    Uses raw attention scores s_ij (pre-softmax) for visualization.
    Post-softmax alpha_ij sums to 1 per TBM node, making C_sum
    uninformative; raw scores preserve absolute magnitude differences.

    The primary output C_mean is degree-normalised: it divides the sum of
    incident softplus-transformed raw scores by the number of incident
    edges, yielding the average per-edge relevance.  The softplus
    non-negative transform is applied because raw pre-softmax scores can
    be negative, and direct summation would allow positive and negative
    scores to cancel.  Softplus preserves the monotonic ordering of raw
    scores while ensuring non-negativity.  Degree normalisation prevents
    nodes with more candidate edges from receiving artificially inflated
    relevance solely due to edge count.

    Args:
      attention: raw attention scores s_ij (pre-softmax), shape (E,)
      edge_index: edge index tensor, shape (2, E)
      n_tbm_nodes: number of TBM surface nodes
      epsilon: small constant to avoid division by zero

    Returns:
      (C_mean, C_sum): tuple of (degree-normalised, raw-sum) attention.
        - C_mean: average softplus(score) per incident edge (degree-normalised)
        - C_sum: total softplus(score) per TBM node (raw sum, for diagnostics)
        Both are (n_tbm_nodes,) numpy arrays.
    """
    dst = edge_index[1].cpu().numpy()
    att = attention.detach().cpu().numpy()

    # Apply softplus non-negative transform to handle negative raw scores
    att_nonneg = np.log1p(np.exp(att))

    C_sum = np.zeros(n_tbm_nodes, dtype=np.float32)
    C_count = np.zeros(n_tbm_nodes, dtype=np.float32)

    for k in range(len(dst)):
        j = dst[k]
        C_sum[j] += att_nonneg[k]
        C_count[j] += 1

    # Degree-normalised: average per-edge relevance
    C_mean = C_sum / (C_count + epsilon)
    return C_mean, C_sum


def plot_shield_hotspot(tbm_positions: np.ndarray, tbm_components: np.ndarray,
                        C_j: np.ndarray, save_path: Optional[str] = None,
                        component_boundaries: Optional[list] = None):
    """Plot attention-derived relevance as an unwrapped TBM surface map.

    Uses cividis colormap (colorblind-safe) and marks geological orientation
    (crown, invert, springline) and TBM component boundaries.
    """
    shield_mask = tbm_components >= 1
    pos = tbm_positions[shield_mask]
    C_j_arr = np.asarray(C_j)
    # Handle multi-dimensional C_j (e.g., (T, N) for time series)
    if C_j_arr.ndim > 1:
        C_j_arr = C_j_arr[-1]  # take last time step
    C = C_j_arr[shield_mask]

    if len(pos) == 0:
        return

    apply_publication_style()
    x, theta = compute_unwrapped_coords(pos)

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    setup_unwrapped_surface_axes(ax, x_range=(x.min(), x.max()),
                                 component_boundaries=component_boundaries)
    vmax = np.percentile(C, 95) if len(C) else 1.0
    scatter = ax.scatter(
        x, theta, c=C, cmap=IJGIS_CMAPS["sequential"], s=2, alpha=0.75,
        vmin=0, vmax=vmax,
    )
    ax.set_title("Shield surface response-supervised hotspot")
    plt.colorbar(scatter, ax=ax, label=r"$C_j$ (learned relevance)")
    fig.tight_layout()
    save_figure(fig, save_path)


def plot_hotspot_vs_response(chainages: list, C_mean: np.ndarray,
                             monitoring: np.ndarray, var_names: list,
                             geo_chainages: Optional[np.ndarray] = None,
                             geo_values: Optional[np.ndarray] = None,
                             geo_label: str = "TSP velocity",
                             save_path: Optional[str] = None):
    """Plot mean hotspot relevance with aligned monitoring responses and geological annotation.

    Args:
      geo_chainages: chainage positions of geological data (e.g., TSP velocity anomalies).
      geo_values: geological attribute values at those chainages.
      geo_label: label for the geological annotation axis.
    """
    apply_publication_style()
    n_geo = 1 if (geo_chainages is not None and geo_values is not None) else 0
    n_vars = len(var_names)
    n_rows = n_vars + 1 + n_geo
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(7.2, 1.25 * n_rows), sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(chainages, C_mean, color="#6A3D9A", linewidth=1.2)
    axes[0].set_ylabel(r"Mean $C_j$")
    axes[0].grid(True)
    axes[0].set_title("Hotspot relevance, geological context, and monitoring response")

    row = 1
    # Geological annotation (e.g., TSP velocity anomalies)
    if n_geo:
        ax_geo = axes[row]
        ax_geo.plot(geo_chainages, geo_values, color="#33A02C", linewidth=1.0, alpha=0.8)
        ax_geo.set_ylabel(geo_label)
        ax_geo.grid(True)
        # Mark low-velocity anomalies (potential weak zones)
        if len(geo_values) > 0:
            threshold = np.percentile(geo_values, 25)
            ax_geo.axhline(y=threshold, color="#E31A1C", linestyle="--", linewidth=0.7, alpha=0.6)
            ax_geo.fill_between(geo_chainages, geo_values, threshold,
                                where=geo_values < threshold,
                                color="#E31A1C", alpha=0.15, label="Low-velocity zone")
            ax_geo.legend(fontsize=7, loc="upper right")
        row += 1

    for i, name in enumerate(var_names):
        ax = axes[row + i]
        ax.plot(chainages, monitoring[:, i], color="#1B6CA8", linewidth=1.0)
        ax.set_ylabel(name)
        ax.grid(True)

    axes[-1].set_xlabel("Chainage (m)")
    fig.tight_layout()
    save_figure(fig, save_path)
