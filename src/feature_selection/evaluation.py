"""
Part 4 -- Downstream Evaluation for Feature Selection.

Trains and evaluates the SAME downstream classifier on different
feature subsets for fair comparison.

Leakage prevention:
    - Feature selection is fitted ONLY on X_train
    - Downstream classifier is trained ONLY on X_train
    - Validation metrics are computed on X_val
    - Test set is UNTOUCHED during selection/tuning
"""
from __future__ import annotations

import time
import warnings
import numpy as np
from typing import Dict, Any, List

warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import compute_metrics, print_metrics
from src.feature_selection.config import SEED, DOWNSTREAM_CLASSIFIER


def get_downstream_classifier(name: str = DOWNSTREAM_CLASSIFIER, seed: int = SEED):
    """
    Return a fresh instance of the controlled downstream classifier.
    The same classifier is used for ALL feature-subset comparisons.
    """
    if name == "SVM_RBF":
        return SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=seed
        )
    elif name == "LogReg":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
    else:
        raise ValueError(f"Unknown downstream classifier: {name}")


def evaluate_feature_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_indices: List[int],
    method_name: str,
    k: int,
    classifier_name: str = DOWNSTREAM_CLASSIFIER,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Train downstream classifier on selected features and evaluate on validation set.

    Returns:
        Dictionary with full evaluation metrics + metadata.
    """
    # Select features
    X_tr_sel = X_train[:, feature_indices]
    X_va_sel = X_val[:, feature_indices]

    assert X_tr_sel.shape[1] == len(feature_indices)
    assert X_va_sel.shape[1] == len(feature_indices)

    # Train
    clf = get_downstream_classifier(classifier_name, seed)
    t0 = time.time()
    clf.fit(X_tr_sel, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = clf.predict(X_va_sel)
    inference_time = time.time() - t0

    # Probabilities
    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_va_sel)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_prob = clf.decision_function(X_va_sel)

    # Metrics
    model_label = f"{method_name}_k{k}"
    metrics = compute_metrics(y_val, y_pred, y_prob, model_label)
    metrics["method"] = method_name
    metrics["feature_count"] = k
    metrics["selected_features"] = sorted(feature_indices)
    metrics["classifier"] = classifier_name
    metrics["training_time_s"] = round(train_time, 6)
    metrics["inference_time_s"] = round(inference_time, 6)
    metrics["seed"] = seed

    return metrics


def evaluate_all_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classifier_name: str = DOWNSTREAM_CLASSIFIER,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Evaluate with ALL features as baseline.
    """
    all_indices = list(range(X_train.shape[1]))
    metrics = evaluate_feature_subset(
        X_train, y_train, X_val, y_val,
        feature_indices=all_indices,
        method_name="all_features",
        k=X_train.shape[1],
        classifier_name=classifier_name,
        seed=seed,
    )
    return metrics


def run_comparison_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classical_results: Dict,
    qaoa_results: Dict,
    budgets: List[int],
    classifier_name: str = DOWNSTREAM_CLASSIFIER,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """
    Run fair comparison: same classifier on all feature subsets.

    Evaluates:
        1. All features (baseline)
        2. Each classical method at each budget
        3. QAOA (at its naturally selected budget)
    """
    all_metrics = []

    # 1. All features baseline
    print("\n--- Evaluating: All Features (baseline) ---")
    baseline = evaluate_all_features(
        X_train, y_train, X_val, y_val, classifier_name, seed
    )
    print_metrics(baseline)
    all_metrics.append(baseline)

    # 2. Classical methods
    for method_name, budget_results in classical_results.items():
        for k, result in budget_results.items():
            k_int = int(k) if isinstance(k, str) else k
            print(f"\n--- Evaluating: {method_name} k={k_int} ---")
            metrics = evaluate_feature_subset(
                X_train, y_train, X_val, y_val,
                feature_indices=result["selected_indices"],
                method_name=method_name,
                k=k_int,
                classifier_name=classifier_name,
                seed=seed,
            )
            print_metrics(metrics)
            all_metrics.append(metrics)

    # 3. QAOA
    if isinstance(qaoa_results, dict) and "selected_indices" in qaoa_results:
        qaoa_list = [qaoa_results]
    elif isinstance(qaoa_results, list):
        qaoa_list = qaoa_results
    else:
        qaoa_list = []

    for qaoa_r in qaoa_list:
        k_qaoa = len(qaoa_r["selected_indices"])
        tag = qaoa_r.get("tag", "")
        method_label = f"qaoa_{tag}" if tag else "qaoa"
        print(f"\n--- Evaluating: QAOA k={k_qaoa} {tag} ---")
        metrics = evaluate_feature_subset(
            X_train, y_train, X_val, y_val,
            feature_indices=qaoa_r["selected_indices"],
            method_name=method_label,
            k=k_qaoa,
            classifier_name=classifier_name,
            seed=seed,
        )
        # Add QAOA-specific metadata
        metrics["qaoa_objective"] = qaoa_r.get("objective_value")
        metrics["qaoa_layers"] = qaoa_r.get("qaoa_layers")
        metrics["qaoa_shots"] = qaoa_r.get("shots")
        metrics["qaoa_time_s"] = qaoa_r.get("total_time_s")
        metrics["n_qubits"] = qaoa_r.get("n_qubits")
        print_metrics(metrics)
        all_metrics.append(metrics)

    return all_metrics
