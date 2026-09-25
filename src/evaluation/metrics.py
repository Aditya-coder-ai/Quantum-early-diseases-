"""
Evaluation utilities for all models in the MindMatrix pipeline.

Provides consistent metric computation, confusion matrix generation,
and result formatting for both classical and quantum models.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import RESULTS_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "model"
) -> dict:
    """
    Compute comprehensive binary classification metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities for positive class (optional)
        model_name: Name identifier for the model
        
    Returns:
        Dictionary of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(specificity, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    
    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print(f"\n--- {metrics['model']} ---")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"  Confusion:   TP={metrics['tp']} FP={metrics['fp']} "
          f"TN={metrics['tn']} FN={metrics['fn']}")
    print(f"  FNR:         {metrics['false_negative_rate']:.4f}")


def save_metrics(metrics_list: list[dict], filename: str = "results.csv"):
    """Save a list of metric dicts to CSV."""
    df = pd.DataFrame(metrics_list)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[EVAL] Results saved to {path}")
    return path


def get_roc_curve_data(y_true, y_prob):
    """Compute ROC curve data points."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def get_pr_curve_data(y_true, y_prob):
    """Compute precision-recall curve data points."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist()
    }
