"""
Part 6: Publication-Quality Visualization Suite for VQC Experiments.
Generates research figures for training curves, comparative bar charts,
circuit depth sweeps, feature count scaling, confusion matrices, and ROC/PR curves.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    """Plot training and validation loss, accuracy, PR-AUC, and minority recall over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0, 0].set_title("Loss vs. Epoch", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Weighted BCE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy
    axes[0, 1].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="#2ca02c", lw=2)
    axes[0, 1].set_title("Validation Accuracy vs. Epoch", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Validation PR-AUC
    axes[1, 0].plot(epochs, history["val_pr_auc"], label="Val PR-AUC", color="#d62728", lw=2)
    axes[1, 0].set_title("Validation PR-AUC vs. Epoch", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("PR-AUC (Minority)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Minority Recall (Sensitivity)
    axes[1, 1].plot(epochs, history["val_minority_recall"], label="Val Minority Recall", color="#9467bd", lw=2)
    axes[1, 1].set_title("Validation Minority Recall (Sensitivity) vs. Epoch", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Minority Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_vqc_vs_classical_comparison(
    comparison_records: List[Dict[str, Any]],
    output_path: str
) -> None:
    """Plot comparative grouped bar chart for VQC vs Classical baselines."""
    methods = [r["method"] for r in comparison_records]
    metrics = ["accuracy", "minority_recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Minority Recall", "F1 Score", "ROC-AUC", "PR-AUC"]

    x = np.arange(len(methods))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        values = [r.get(m, 0.0) for r in comparison_records]
        ax.bar(x + i * width, values, width, label=label, color=colors[i], alpha=0.9)

    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Classical vs. Variational Quantum Classifier (VQC) Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax.set_ylim([0.7, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_circuit_depth_scaling(depth_records: List[Dict[str, Any]], output_path: str) -> None:
    """Plot accuracy, minority recall, and training time across circuit depths."""
    depths = [r["depth"] for r in depth_records]
    accuracies = [r["accuracy"] for r in depth_records]
    recalls = [r["minority_recall"] for r in depth_records]
    times = [r["training_time_s"] for r in depth_records]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    p1 = ax1.plot(depths, accuracies, marker="o", color="#1f77b4", lw=2, label="Accuracy")
    p2 = ax1.plot(depths, recalls, marker="s", color="#2ca02c", lw=2, label="Minority Recall")
    p3 = ax2.plot(depths, times, marker="^", color="#d62728", lw=2, linestyle="--", label="Training Time (s)")

    ax1.set_xlabel("Circuit Depth (Variational Layers)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Training Time (seconds)", fontsize=11, color="#d62728", fontweight="bold")
    ax1.set_xticks(depths)
    ax1.set_ylim([0.8, 1.02])

    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")
    plt.title("VQC Performance and Runtime vs. Circuit Depth", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_count_scaling(feature_records: List[Dict[str, Any]], output_path: str) -> None:
    """Plot VQC performance scaling as feature/qubit count increases."""
    counts = [r["feature_count"] for r in feature_records]
    accuracies = [r["accuracy"] for r in feature_records]
    recalls = [r["minority_recall"] for r in feature_records]
    pr_aucs = [r["pr_auc"] for r in feature_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, accuracies, marker="o", lw=2, label="Accuracy", color="#1f77b4")
    ax.plot(counts, recalls, marker="s", lw=2, label="Minority Recall", color="#2ca02c")
    ax.plot(counts, pr_aucs, marker="^", lw=2, label="PR-AUC", color="#ff7f0e")

    ax.set_xlabel("Feature Count / Qubit Count (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_xticks(counts)
    ax.set_ylim([0.8, 1.02])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title("VQC Classification Performance vs. Qubit / Feature Count", fontsize=12, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix_vqc(cm: np.ndarray, output_path: str, title: str = "VQC Test Confusion Matrix") -> None:
    """Plot confusion matrix heatmap with clinical annotations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticklabels(["", "Malignant (0)", "Benign (1)"], fontsize=10)
    ax.set_yticklabels(["", "Malignant (0)", "Benign (1)"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    output_path: str,
    model_name: str = "VQC"
) -> None:
    """Plot Receiver Operating Characteristic and Precision-Recall curves on test set."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_score = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{model_name} (AUC = {roc_score:.4f})")
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Precision-Recall curve (for minority class: label 0)
    y_minority = 1 - y_true
    minority_probs = 1.0 - probs
    prec, rec, _ = precision_recall_curve(y_minority, minority_probs)
    pr_score = auc(rec, prec)
    ax2.plot(rec, prec, color="#d62728", lw=2, label=f"{model_name} Minority (PR-AUC = {pr_score:.4f})")
    ax2.set_xlabel("Recall / Sensitivity", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Minority Class Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
