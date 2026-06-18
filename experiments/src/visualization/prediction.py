"""Publication-quality prediction result visualizations."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import PALETTE, add_panel_label, apply_publication_style, save_figure


def _series_color(name: str, idx: int) -> str:
    lowered = name.lower()
    if "full" in lowered:
        return PALETTE["full"]
    if "tsp" in lowered and "lstm" in lowered:
        return PALETTE["tsp_lstm"]
    if "lstm" in lowered:
        return PALETTE["lstm"]
    if "xgboost" in lowered:
        return PALETTE["baseline"]
    if "persist" in lowered:
        return PALETTE["persistence"]
    palette = [PALETTE["accent"], "#6C5B7B", "#355C7D", "#E9C46A"]
    return palette[idx % len(palette)]


def plot_prediction_comparison(chainages: np.ndarray,
                               y_true: np.ndarray,
                               predictions: dict,
                               var_idx: int = 0,
                               var_name: str = "Variable",
                               save_path: Optional[str] = None):
    """Plot prediction trajectories with a residual panel."""
    apply_publication_style()
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8.6, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.0]}
    )

    truth = y_true[:, var_idx]
    ax_main.plot(
        chainages, truth,
        color=PALETTE["truth"], linewidth=2.0, label="Observed", zorder=3,
    )

    for i, (name, pred) in enumerate(predictions.items()):
        series = pred[:, var_idx]
        mae = np.mean(np.abs(series - truth))
        color = _series_color(name, i)
        ax_main.plot(
            chainages, series,
            color=color, linewidth=1.8, linestyle="--",
            label=f"{name} (MAE={mae:.3f})", zorder=2,
        )
        ax_res.plot(
            chainages, series - truth,
            color=color, linewidth=1.2, alpha=0.95,
        )

    ax_main.set_ylabel(var_name)
    ax_main.grid(True)
    add_panel_label(ax_main, "A")

    ax_res.axhline(0.0, color="#666666", linewidth=0.9, linestyle=":")
    ax_res.set_xlabel("Chainage (m)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True)
    add_panel_label(ax_res, "B")

    handles, labels = ax_main.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"Prediction Comparison for {var_name}", y=0.93, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, save_path)


def plot_metrics_bar(metrics_dict: dict, metric_name: str = "rmse",
                     save_path: Optional[str] = None):
    """Plot model comparison bars with publication styling."""
    apply_publication_style()
    models = list(metrics_dict.keys())
    values = [m[metric_name] for m in metrics_dict.values()]
    colors = [_series_color(name, idx) for idx, name in enumerate(models)]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.barh(models, values, color=colors, alpha=0.88)
    ax.set_xlabel(metric_name.upper())
    ax.set_title(f"Model Comparison by {metric_name.upper()}")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=4)
    ax.invert_yaxis()
    ax.grid(True, axis="x")

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_ablation_results(ablation_results: dict, metric_name: str = "rmse",
                          save_path: Optional[str] = None):
    """Plot ablation results with consistent styling."""
    apply_publication_style()
    names = list(ablation_results.keys())
    values = [ablation_results[n][metric_name] for n in names]
    colors = [_series_color(name, idx) for idx, name in enumerate(names)]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bars = ax.bar(names, values, color=colors, alpha=0.9)
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"Ablation Study on {metric_name.upper()}")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.grid(True, axis="y")
    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()
    save_figure(fig, save_path)
