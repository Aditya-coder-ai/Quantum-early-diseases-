"""
Part 8: Model-Independent & Model-Based Feature Importance Baselines.
Implements Permutation Feature Importance and Classical Model-Based Attributions
as rigorous benchmarks to contrast with SHAP values.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Union
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


class PermutationImportanceExplainer:
    """
    Model-independent Permutation Feature Importance.
    Measures the increase in prediction error (or drop in metric) when values
    of a specific feature are randomly permuted, breaking its relationship with the target.
    """
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        metric: str = "roc_auc",
        n_repeats: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            predict_fn: Callable accepting (N, D) and returning 1D probability of positive class (Malignant).
            metric: 'roc_auc', 'accuracy', or 'log_loss'.
            n_repeats: Number of permutation iterations per feature.
            seed: Random seed for reproducible shuffling.
        """
        self.predict_fn = predict_fn
        self.metric = metric
        self.n_repeats = n_repeats
        self.seed = seed

    def _eval_metric(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if self.metric == "roc_auc":
            try:
                return float(roc_auc_score(y_true, y_prob))
            except Exception:
                return 0.5
        elif self.metric == "accuracy":
            preds = (y_prob >= 0.5).astype(int)
            return float(accuracy_score(y_true, preds))
        elif self.metric == "log_loss":
            eps = 1e-15
            clipped_prob = np.clip(y_prob, eps, 1.0 - eps)
            return -float(log_loss(y_true, clipped_prob))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def compute_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Computes mean metric drop across n_repeats for each feature.
        
        Returns:
            Dict mapping feature name -> mean importance drop.
        """
        rng = np.random.RandomState(self.seed)
        X_arr = np.asarray(X, dtype=np.float64).copy()
        y_arr = np.asarray(y).ravel()
        n_samples, n_features = X_arr.shape

        names = feature_names or [f"feat_{i}" for i in range(n_features)]

        # Baseline performance on intact data
        base_probs = self.predict_fn(X_arr)
        base_score = self._eval_metric(y_arr, base_probs)

        importance_scores = {}

        for j in range(n_features):
            drops = []
            for _ in range(self.n_repeats):
                X_perm = X_arr.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_probs = self.predict_fn(X_perm)
                perm_score = self._eval_metric(y_arr, perm_probs)
                # Performance drop indicates feature importance
                drops.append(max(0.0, base_score - perm_score))
            importance_scores[names[j]] = round(float(np.mean(drops)), 5)

        return importance_scores


def extract_classical_model_importance(
    model: Any,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Extracts native model-based feature importance from classical estimators
    (e.g., LogisticRegression coefficients, Random Forest MDI, or Linear SVM weights).
    """
    importance = {}
    if hasattr(model, "coef_"):
        # Linear models: absolute coefficient magnitude
        weights = np.abs(model.coef_).ravel()
        total = weights.sum()
        norm_weights = weights / (total if total > 0 else 1.0)
        for name, w in zip(feature_names, norm_weights):
            importance[name] = round(float(w), 5)
    elif hasattr(model, "feature_importances_"):
        # Tree-based ensembles
        weights = model.feature_importances_
        for name, w in zip(feature_names, weights):
            importance[name] = round(float(w), 5)
    else:
        # Default uniform
        uniform_val = 1.0 / len(feature_names)
        for name in feature_names:
            importance[name] = round(uniform_val, 5)

    return importance
