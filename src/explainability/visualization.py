"""
Part 8: Publication-Quality Interpretability Visualizations.
Generates diagnostic plots: Global SHAP Bars, Local Patient Waterfall Plots,
QAOA Selection vs. SHAP Comparisons, Faithfulness Curves, and Grad-CAM Overlays.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.explainability.schemas import LocalExplanation, GlobalFeatureImportance


def plot_global_shap_bar(
    rankings: List[GlobalFeatureImportance],
    save_path: str,
    title: str = "Global Feature Attribution (Mean |SHAP|)"
) -> None:
    """Plots horizontal bar chart of global mean absolute SHAP values."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    names = [r.feature_name for r in reversed(rankings)]
    vals = [r.mean_abs_shap for r in reversed(rankings)]
    colors = ["#d9534f" if r.direction_trend == "positive_risk" else "#5bc0de" if r.direction_trend == "protective" else "#f0ad4e" for r in reversed(rankings)]

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.45)), dpi=300)
    bars = ax.barh(names, vals, color=colors, edgecolor="black", alpha=0.85)

    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Malignancy Probability)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_local_waterfall(
    local_exp: LocalExplanation,
    save_path: str,
    max_display: int = 8
) -> None:
    """
    Renders patient-specific directional attribution bar chart.
    Red/Orange bars increase malignancy risk; Blue bars decrease risk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    names = local_exp.feature_names
    shap_vals = local_exp.shap_values
    feat_vals = local_exp.feature_values

    # Pair and sort by absolute contribution
    pairs = sorted(zip(names, shap_vals, feat_vals), key=lambda x: abs(x[1]), reverse=True)[:max_display]
    pairs = list(reversed(pairs))

    plot_names = [f"{p[0]} = {p[2]:.2f}" for p in pairs]
    plot_shaps = [p[1] for p in pairs]
    colors = ["#d9534f" if v > 0 else "#2b8a3e" for v in plot_shaps]

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(plot_names) * 0.45)), dpi=300)
    bars = ax.barh(plot_names, plot_shaps, color=colors, edgecolor="black", alpha=0.85)

    ax.axvline(0, color="black", linestyle="-", linewidth=1.2)
    ax.set_xlabel("SHAP Attribution (Change in Malignancy Probability)", fontsize=11, fontweight="bold")
    subtitle = (
        f"Patient {local_exp.sample_id} | Diagnosis: {local_exp.class_label} | "
        f"Malignancy Risk: {local_exp.malignant_probability * 100:.1f}%\n"
        f"Base Rate: {local_exp.base_value * 100:.1f}% | Risk Level: {local_exp.risk_assessment}"
    )
    ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate positive/negative shift
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (0.005 if w >= 0 else -0.015)
        ha = "left" if w >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{w:+.3f}",
                va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_qaoa_vs_shap_comparison(
    comparison_df: pd.DataFrame,
    save_path: str
) -> None:
    """
    Contrasts QAOA selection ranking against SHAP prediction importance.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(comparison_df))
    width = 0.35

    # Reverse ranks so higher bar means higher priority (Rank 1 -> value 8)
    max_r = len(comparison_df)
    qaoa_priority = [max_r - (r or max_r) + 1 for r in comparison_df["qaoa_selection_rank"]]
    shap_priority = [max_r - (r or max_r) + 1 for r in comparison_df["shap_rank"]]

    ax.bar(x - width / 2, qaoa_priority, width, label="QAOA Selection Priority", color="#6f42c1", edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, shap_priority, width, label="SHAP Attribution Priority", color="#fd7e14", edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["feature"], rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Relative Priority (Higher is More Critical)", fontsize=11, fontweight="bold")
    ax.set_title("QAOA Optimization Selection vs. SHAP Model Attribution", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_faithfulness_comparison(
    faithfulness_metrics: List[Any],
    save_path: str
) -> None:
    """
    Visualizes the prediction shift when top-attributed vs least-attributed features are perturbed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    top_deltas = [m.top_k_delta for m in faithfulness_metrics]
    least_deltas = [m.least_k_delta for m in faithfulness_metrics]
    sample_ids = [str(m.sample_id) for m in faithfulness_metrics]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    x = np.arange(len(sample_ids))
    width = 0.35

    ax.bar(x - width / 2, top_deltas, width, label="|ΔP| Perturbing Top Features", color="#d9534f", edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, least_deltas, width, label="|ΔP| Perturbing Least Features", color="#6c757d", edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Patient {s}" for s in sample_ids], fontsize=10)
    ax.set_ylabel("|Probability Shift| (|ΔP|)", fontsize=11, fontweight="bold")
    ax.set_title("Faithfulness Verification: Feature Perturbation Response", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_gradcam_overlay(
    image_2d: np.ndarray,
    heatmap_2d: np.ndarray,
    save_path: str,
    title: str = "Grad-CAM Spatial Feature Activation"
) -> None:
    """Renders 3-panel Grad-CAM visualization: Raw Image, Heatmap, and Blended Overlay."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # 1. Raw image
    axes[0].imshow(image_2d, cmap="gray")
    axes[0].set_title("Input Medical Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Heatmap
    im1 = axes[1].imshow(heatmap_2d, cmap="jet")
    axes[1].set_title("Grad-CAM Class Activation", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Blended Overlay
    axes[2].imshow(image_2d, cmap="gray")
    axes[2].imshow(heatmap_2d, cmap="jet", alpha=0.45)
    axes[2].set_title("Aligned Diagnostic Overlay", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
