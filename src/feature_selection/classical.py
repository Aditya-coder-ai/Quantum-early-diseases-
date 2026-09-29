"""
Part 4 — Classical Feature Selection Methods.

Implements Mutual Information and RFE (with additional LASSO and RF
importance for completeness).

All methods are fitted ONLY on (X_train, y_train) to prevent leakage.
"""
from __future__ import annotations

import json
import os
import time
import logging
import numpy as np
from typing import Dict, List, Any

from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif,
    RFE
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.feature_selection.config import (
    SEED, ARTIFACT_DIRS, FEATURE_BUDGETS, INPUT_FEATURE_DIM,
)
from src.feature_selection.registry import (
    FeatureEntry, get_selected_names
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Individual Selectors
# ══════════════════════════════════════════════════════════════════════

def select_mutual_info(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Select top-k features by mutual information (fitted on training data).

    Returns dict with selected indices, scores, and timing.
    """
    t0 = time.time()
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(
            X, y, discrete_features=False, random_state=seed, n_neighbors=5
        ),
        k=k,
    )
    selector.fit(X_train, y_train)
    elapsed = time.time() - t0

    indices = selector.get_support(indices=True).tolist()
    scores = selector.scores_.tolist()

    assert len(indices) == k, f"Expected {k} features, got {len(indices)}"
    assert all(0 <= i < X_train.shape[1] for i in indices), "Invalid feature index"

    return {
        "method": "mutual_info",
        "k": k,
        "selected_indices": sorted(indices),
        "scores": scores,
        "selection_time_s": round(elapsed, 4),
        "seed": seed,
    }


def select_rfe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Recursive Feature Elimination with a linear SVM estimator.
    Fitted on training data only.
    """
    t0 = time.time()
    estimator = SVC(kernel="linear", random_state=seed)
    rfe = RFE(estimator, n_features_to_select=k, step=1)
    rfe.fit(X_train, y_train)
    elapsed = time.time() - t0

    indices = np.where(rfe.support_)[0].tolist()
    rankings = rfe.ranking_.tolist()

    assert len(indices) == k, f"Expected {k} features, got {len(indices)}"
    assert all(0 <= i < X_train.shape[1] for i in indices), "Invalid feature index"

    return {
        "method": "rfe",
        "k": k,
        "selected_indices": sorted(indices),
        "rankings": rankings,
        "selection_time_s": round(elapsed, 4),
        "seed": seed,
    }


def select_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    L1/LASSO feature selection — selects top-k features by absolute
    coefficient magnitude from cross-validated LASSO.
    """
    t0 = time.time()
    lasso = LassoCV(cv=3, random_state=seed, max_iter=5000)
    lasso.fit(X_train, y_train)
    elapsed = time.time() - t0

    coef_abs = np.abs(lasso.coef_)
    top_k_indices = np.argsort(coef_abs)[::-1][:k].tolist()

    assert len(top_k_indices) == k
    assert all(0 <= i < X_train.shape[1] for i in top_k_indices)

    return {
        "method": "lasso",
        "k": k,
        "selected_indices": sorted(top_k_indices),
        "coefficients": coef_abs.tolist(),
        "best_alpha": float(lasso.alpha_),
        "selection_time_s": round(elapsed, 4),
        "seed": seed,
    }


def select_rf_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Random Forest importance-based selection.
    """
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0

    importances = rf.feature_importances_
    top_k_indices = np.argsort(importances)[::-1][:k].tolist()

    assert len(top_k_indices) == k
    assert all(0 <= i < X_train.shape[1] for i in top_k_indices)

    return {
        "method": "rf_importance",
        "k": k,
        "selected_indices": sorted(top_k_indices),
        "importances": importances.tolist(),
        "selection_time_s": round(elapsed, 4),
        "seed": seed,
    }


# ══════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════

SELECTOR_MAP = {
    "mutual_info": select_mutual_info,
    "rfe": select_rfe,
    "lasso": select_lasso,
    "rf_importance": select_rf_importance,
}


def run_classical_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    methods: List[str],
    budgets: List[int],
    registry: List[FeatureEntry],
    seed: int = SEED,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Run multiple classical feature-selection methods at multiple budgets.

    Returns:
        nested dict: results[method_name][k] = selection_result_dict
    """
    D = X_train.shape[1]
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for method_name in methods:
        if method_name not in SELECTOR_MAP:
            raise ValueError(f"Unknown method: {method_name}")
        selector_fn = SELECTOR_MAP[method_name]
        results[method_name] = {}

        for k in budgets:
            if k > D:
                logger.warning("Budget k=%d exceeds D=%d, skipping", k, D)
                continue

            print(f"  [Classical] {method_name} k={k} ...", end=" ", flush=True)
            result = selector_fn(X_train, y_train, k=k, seed=seed)

            # Attach feature names from registry
            result["selected_feature_names"] = get_selected_names(
                registry, result["selected_indices"]
            )

            results[method_name][k] = result
            print(
                f"-> {result['selected_indices']}  "
                f"({result['selection_time_s']:.2f}s)",
                flush=True,
            )

    return results


def save_classical_results(
    results: Dict[str, Dict[int, Dict[str, Any]]]
) -> List[str]:
    """Save classical selection results per method."""
    saved = []
    for method_name, budget_results in results.items():
        out_dir = ARTIFACT_DIRS.get(method_name, ARTIFACT_DIRS["comparison"])
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"selection_results.json")
        with open(path, "w") as f:
            # Convert int keys to str for JSON
            json.dump(
                {str(k): v for k, v in budget_results.items()},
                f, indent=2,
            )
        saved.append(path)
    return saved
