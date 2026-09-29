"""
Comprehensive evaluation metrics, threshold tuning, and resource profiling for Part 9.
"""
from __future__ import annotations

import time
import os
import psutil
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    disease_class: int = 0,
    model_name: str = "model",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes rigorous clinical metrics emphasizing minority disease detection (Class 0 = Malignant).
    
    Args:
        y_true: Ground truth labels (0=Malignant, 1=Benign in WDBC convention)
        y_pred: Predicted labels (0/1)
        y_probs: Predicted probability of Class 1 (Benign). If probability of Malignant is needed,
                 p_malignant = 1.0 - y_probs.
        disease_class: Label corresponding to the target pathology (default 0 for WDBC)
        model_name: Model identifier string
        threshold: Decision threshold applied to derive y_pred
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    
    # Binary indicators for disease class (1 if Malignant, 0 if Benign)
    y_dis_true = (y_true == disease_class).astype(int)
    y_dis_pred = (y_pred == disease_class).astype(int)
    
    # Confusion matrix with disease as positive
    cm_dis = confusion_matrix(y_dis_true, y_dis_pred, labels=[0, 1])
    # labels=[0, 1] -> row 0: true benign, row 1: true malignant
    # col 0: pred benign, col 1: pred malignant
    tn = int(cm_dis[0, 0])  # Benign predicted Benign
    fp = int(cm_dis[0, 1])  # Benign predicted Malignant
    fn = int(cm_dis[1, 0])  # Malignant predicted Benign (CRITICAL FALSE NEGATIVE)
    tp = int(cm_dis[1, 1])  # Malignant predicted Malignant

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    # Disease-specific (Malignant) metrics
    rec_dis = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec_dis = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    spec_dis = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv_dis = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1_dis = float(2 * prec_dis * rec_dis / (prec_dis + rec_dis)) if (prec_dis + rec_dis) > 0 else 0.0
    fnr_dis = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr_dis = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "threshold": round(float(threshold), 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "mcc": round(mcc, 4),
        # Disease (Malignant) class metrics
        "recall": round(rec_dis, 4),          # Clinical Sensitivity
        "sensitivity": round(rec_dis, 4),      # Alias for recall
        "specificity": round(spec_dis, 4),    # Clinical Specificity
        "precision": round(prec_dis, 4),      # Positive Predictive Value (PPV)
        "npv": round(npv_dis, 4),              # Negative Predictive Value
        "f1": round(f1_dis, 4),
        "fnr": round(fnr_dis, 4),              # False Negative Rate
        "fpr": round(fpr_dis, 4),              # False Positive Rate
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_negatives": fn,                 # Highlight critical false negatives
    }

    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=float).ravel()
        # Probability of disease (Malignant)
        p_dis = 1.0 - y_probs if (y_probs <= 1.0).all() and (y_probs >= 0.0).all() else y_probs
        
        try:
            roc_auc = float(roc_auc_score(y_dis_true, p_dis))
        except Exception:
            roc_auc = 0.5
            
        try:
            pr_auc = float(average_precision_score(y_dis_true, p_dis))
        except Exception:
            pr_auc = 0.0
            
        try:
            brier = float(brier_score_loss(y_dis_true, p_dis))
        except Exception:
            brier = 1.0

        metrics["roc_auc"] = round(roc_auc, 4)
        metrics["pr_auc"] = round(pr_auc, 4)
        metrics["brier_score"] = round(brier, 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None

    return metrics


def tune_decision_threshold(
    y_val: np.ndarray,
    y_val_probs: np.ndarray,
    metric: str = "f1",
    disease_class: int = 0,
    min_th: float = 0.1,
    max_th: float = 0.9,
    n_steps: int = 41
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Finds the optimal decision threshold strictly on the validation set.
    Never tunes on the test set.
    
    Returns:
        best_threshold, best_metrics, threshold_sweep_history
    """
    y_val = np.asarray(y_val, dtype=int).ravel()
    y_val_probs = np.asarray(y_val_probs, dtype=float).ravel()
    
    thresholds = np.linspace(min_th, max_th, n_steps)
    history = []
    
    best_th = 0.5
    best_score = -1.0
    best_metrics: Dict[str, Any] = {}

    for th in thresholds:
        # If prob of benign < th, predict malignant (0), else benign (1)
        preds = (y_val_probs >= th).astype(int)
        m = compute_extended_metrics(
            y_true=y_val,
            y_pred=preds,
            y_probs=y_val_probs,
            disease_class=disease_class,
            threshold=th,
        )
        
        score = m.get(metric, m["f1"])
        m["sweep_score"] = score
        history.append(m)
        
        if score > best_score:
            best_score = score
            best_th = float(th)
            best_metrics = m

    return best_th, best_metrics, history


def compute_calibration_curve_data(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    disease_class: int = 0,
    n_bins: int = 8
) -> Dict[str, List[float]]:
    """Computes empirical calibration curve coordinates for disease class."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_probs = np.asarray(y_probs, dtype=float).ravel()
    
    y_dis_true = (y_true == disease_class).astype(int)
    p_dis = 1.0 - y_probs
    
    prob_true, prob_pred = calibration_curve(y_dis_true, p_dis, n_bins=n_bins, strategy="uniform")
    return {
        "prob_true": [round(float(x), 4) for x in prob_true],
        "prob_pred": [round(float(x), 4) for x in prob_pred]
    }


class ResourceProfiler:
    """Monitors peak RAM, execution duration, and inference latency."""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_mem_mb = 0.0
        self.peak_mem_mb = 0.0
        self.start_time = 0.0
        self.elapsed_s = 0.0

    def start(self) -> None:
        self.start_mem_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_mem_mb = self.start_mem_mb
        self.start_time = time.perf_counter()

    def stop(self) -> Dict[str, float]:
        self.elapsed_s = time.perf_counter() - self.start_time
        current_mem = self.process.memory_info().rss / (1024 * 1024)
        self.peak_mem_mb = max(self.peak_mem_mb, current_mem)
        mem_delta_mb = max(0.0, self.peak_mem_mb - self.start_mem_mb)
        return {
            "elapsed_seconds": round(self.elapsed_s, 4),
            "peak_memory_mb": round(self.peak_mem_mb, 2),
            "memory_delta_mb": round(mem_delta_mb, 2),
        }
