"""
Part 5 — Downstream Classifier Evaluation for Imbalance Strategies.

Evaluates the exact SAME downstream classifier across all imbalance strategies:
    1. Original Imbalanced Training (class_weight=None)
    2. Class Weighted Loss (class_weight="balanced")
    3. SMOTE Augmented Training (various ratios)
    4. QGAN Augmented Training (various ratios)

CRITICAL FAIRNESS & LEAKAGE RULES:
    - Same downstream classifier (SVM RBF)
    - Same training split, same validation split, same test split
    - Augmentation applied ONLY to training data
    - Validation and test splits are NEVER modified or augmented
    - Test set is evaluated ONCE at the end for the top candidate strategies
"""
from __future__ import annotations

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from src.imbalance.config import (
    DOWNSTREAM_CLASSIFIER, SEED, MINORITY_CLASS, MAJORITY_CLASS
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_downstream_classifier(
    class_weight: str | Dict[int, float] | None = None,
    seed: int = SEED,
) -> SVC:
    """
    Return a fresh instance of the controlled downstream classifier.
    Controlled parameters: kernel='rbf', probability=True, random_state=seed.
    """
    return SVC(
        kernel="rbf",
        class_weight=class_weight,
        probability=True,
        random_state=seed,
    )


def evaluate_imbalance_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method_name: str,
    augmentation_ratio: float = 0.0,
    class_weight: str | Dict[int, float] | None = None,
    seed: int = SEED,
    minority_class: int = MINORITY_CLASS,
    majority_class: int = MAJORITY_CLASS,
) -> Dict[str, Any]:
    """
    Train downstream classifier on the given training configuration and
    evaluate on the UNTOUCHED validation set.

    Calculates:
        - Overall classification metrics
        - Minority-class specific metrics (sensitivity/recall, precision, F1)
        - Confusion matrix
        - Training and inference latency
    """
    clf = get_downstream_classifier(class_weight=class_weight, seed=seed)

    # Train
    t0 = time.time()
    clf.fit(X_train, y_train)
    training_time = round(time.time() - t0, 6)

    # Validate
    t0 = time.time()
    y_pred = clf.predict(X_val)
    inference_time = round(time.time() - t0, 6)

    # Probabilities
    y_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None

    # Confusion matrix (for binary 0/1: TN=c00, FP=c01, FN=c10, TP=c11)
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Minority class metrics (class 0 = Malignant in WDBC)
    # When minority is 0:
    #   True Minority = 0, Predicted Minority = 0 -> this is TN in standard 0/1 cm!
    #   True Minority = 0, Predicted Majority = 1 -> this is FP (false negative for disease)!
    if minority_class == 0:
        minority_tp = int(tn)
        minority_fn = int(fp)
        minority_fp = int(fn)
        minority_tn = int(tp)
        minority_actual = minority_tp + minority_fn
        minority_recall = round(minority_tp / minority_actual, 4) if minority_actual > 0 else 0.0
        minority_precision = (
            round(minority_tp / (minority_tp + minority_fp), 4)
            if (minority_tp + minority_fp) > 0 else 0.0
        )
        minority_f1 = (
            round(2 * minority_precision * minority_recall / (minority_precision + minority_recall), 4)
            if (minority_precision + minority_recall) > 0 else 0.0
        )
        # Minority PR-AUC (predicting class 0)
        y_val_min = (y_val == minority_class).astype(int)
        y_prob_min = 1.0 - y_prob if y_prob is not None else None
        minority_pr_auc = (
            round(float(average_precision_score(y_val_min, y_prob_min)), 4)
            if y_prob_min is not None else None
        )
    else:
        minority_recall = round(float(recall_score(y_val, y_pred, pos_label=minority_class, zero_division=0)), 4)
        minority_precision = round(float(precision_score(y_val, y_pred, pos_label=minority_class, zero_division=0)), 4)
        minority_f1 = round(float(f1_score(y_val, y_pred, pos_label=minority_class, zero_division=0)), 4)
        minority_pr_auc = round(float(average_precision_score(y_val, y_prob)), 4) if y_prob is not None else None

    # Overall metrics
    acc = round(float(accuracy_score(y_val, y_pred)), 4)
    prec = round(float(precision_score(y_val, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_val, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_val, y_pred, zero_division=0)), 4)
    spec = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    roc_auc = round(float(roc_auc_score(y_val, y_prob)), 4) if y_prob is not None else None
    pr_auc = round(float(average_precision_score(y_val, y_prob)), 4) if y_prob is not None else None

    # Counts
    classes, counts = np.unique(y_train, return_counts=True)
    train_counts = {int(c): int(cnt) for c, cnt in zip(classes, counts)}

    return {
        "method": method_name,
        "augmentation_ratio": round(augmentation_ratio, 2),
        "train_samples": len(y_train),
        "minority_train_samples": train_counts.get(minority_class, 0),
        "majority_train_samples": train_counts.get(majority_class, 0),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "minority_recall": minority_recall,
        "minority_precision": minority_precision,
        "minority_f1": minority_f1,
        "minority_pr_auc": minority_pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "training_time_s": training_time,
        "inference_time_s": inference_time,
        "seed": seed,
    }


def print_evaluation_summary(result: Dict[str, Any]):
    """Pretty-print evaluation metrics."""
    m = result
    print(
        f"  [{m['method']}] ratio={m['augmentation_ratio']} (N_train={m['train_samples']}) | "
        f"Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | "
        f"Minority Recall: {m['minority_recall']:.4f} | Minority F1: {m['minority_f1']:.4f}",
        flush=True,
    )
