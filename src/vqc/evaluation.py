"""
Part 6: Evaluation Metrics & Classical Reference Comparison.
Computes comprehensive clinical metrics (PR-AUC, Minority Recall, Specificity, ROC-AUC)
for fair 1-to-1 comparisons between VQC and classical baselines.
"""
from __future__ import annotations

import time
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.vqc.model import VariationalQuantumClassifier


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    minority_class: int = 0
) -> Dict[str, Any]:
    """
    Computes overall clinical and minority-specific metrics for binary classification.
    Class 0 = Malignant (Minority), Class 1 = Benign (Majority).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Minority Class Metrics (Class 0: Malignant)
    y_min_true = (y_true == minority_class).astype(int)
    y_min_pred = (y_pred == minority_class).astype(int)
    y_min_probs = 1.0 - y_probs

    min_rec = float(recall_score(y_min_true, y_min_pred, zero_division=0))
    min_prec = float(precision_score(y_min_true, y_min_pred, zero_division=0))
    min_f1 = float(f1_score(y_min_true, y_min_pred, zero_division=0))

    try:
        min_pr_auc = float(average_precision_score(y_min_true, y_min_probs))
    except Exception:
        min_pr_auc = 0.0

    # False negatives in oncology screening: malignant lesions (0) predicted as benign (1)
    # cm[0, 1] is malignant (0) misclassified as benign (1)
    malignant_fn = fp

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "minority_recall": round(min_rec, 4),
        "minority_precision": round(min_prec, 4),
        "minority_f1": round(min_f1, 4),
        "minority_pr_auc": round(min_pr_auc, 4),
        "malignant_false_negatives": malignant_fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_vqc_model(
    model: VariationalQuantumClassifier,
    X_angles: np.ndarray,
    y_true: np.ndarray,
    minority_class: int = 0
) -> Dict[str, Any]:
    """
    Evaluates a trained VQC model on a given dataset split, timing inference latency.
    """
    t0 = time.time()
    probs = model.predict_proba(X_angles)
    inference_time = time.time() - t0
    preds = (probs >= 0.5).astype(int)

    metrics = compute_comprehensive_metrics(y_true, preds, probs, minority_class=minority_class)
    metrics["inference_time_s"] = round(inference_time, 4)
    metrics["latency_per_sample_ms"] = round((inference_time / max(1, len(y_true))) * 1000, 3)
    return metrics


def train_and_evaluate_classical_reference(
    classifier_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    class_weight: Optional[Union[str, Dict[int, float]]] = None,
    seed: int = 42,
    minority_class: int = 0
) -> Dict[str, Any]:
    """
    Trains and evaluates an identical classical baseline model on the exact same feature matrix
    and splits for controlled, fair benchmark comparisons.
    """
    cw_dict = "balanced" if class_weight == "balanced" else class_weight

    if classifier_type == "SVM_RBF":
        clf = SVC(kernel="rbf", probability=True, class_weight=cw_dict, random_state=seed)
    elif classifier_type == "LogisticRegression":
        clf = LogisticRegression(class_weight=cw_dict, random_state=seed, max_iter=1000)
    elif classifier_type == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    t0_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0_train

    t0_inf = time.time()
    probs = clf.predict_proba(X_eval)[:, 1]
    inf_time = time.time() - t0_inf
    preds = clf.predict(X_eval)

    metrics = compute_comprehensive_metrics(y_eval, preds, probs, minority_class=minority_class)
    metrics["classifier_type"] = classifier_type
    metrics["training_time_s"] = round(train_time, 4)
    metrics["inference_time_s"] = round(inf_time, 4)
    metrics["latency_per_sample_ms"] = round((inf_time / max(1, len(y_eval))) * 1000, 3)
    return metrics
