"""
Part 4 — Feature-Selection Objective Function.

Defines the mathematical objective that both classical and quantum
feature selectors aim to optimise.

Objective (to MAXIMISE):
    J(S) = Σ_i∈S  relevance_i
           − α Σ_{i<j, i,j∈S} |corr(i,j)|
           − β (|S| − K)²

where:
    S            = selected feature subset
    relevance_i  = univariate predictive score of feature i
    corr(i,j)    = absolute Pearson correlation between features i and j
    K            = target feature budget
    α            = redundancy penalty weight
    β            = cardinality penalty weight

When feeding into a QUBO we negate to convert to a minimisation problem.

The objective is fitted ONLY on training data to prevent leakage.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.feature_selection.config import (
    SEED, QAOA_PENALTY_LAMBDA, QAOA_CARDINALITY_PENALTY,
    QAOA_IMPORTANCE_WEIGHT, PART4_ARTIFACTS_DIR
)


# ── Feature Importance ───────────────────────────────────────────────

def compute_mutual_information(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """
    Compute MI-based relevance score for each feature (on training data only).

    Returns:
        1-D array of shape (D,) with non-negative MI scores.
    """
    mi = mutual_info_classif(
        X_train, y_train, discrete_features=False, random_state=seed, n_neighbors=5
    )
    return mi


def compute_univariate_auc(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """
    Compute per-feature univariate ROC-AUC via 3-fold CV with LogReg.
    Fitted ONLY on training data.

    Returns:
        1-D array of shape (D,) with AUC scores in [0, 1].
    """
    n_features = X_train.shape[1]
    auc_scores = np.zeros(n_features)
    for i in range(n_features):
        X_single = X_train[:, i : i + 1]
        lr = LogisticRegression(max_iter=500, random_state=seed)
        scores = cross_val_score(lr, X_single, y_train, cv=3, scoring="roc_auc")
        auc_scores[i] = np.mean(scores)
    return auc_scores


# ── Feature Redundancy ───────────────────────────────────────────────

def compute_correlation_matrix(X_train: np.ndarray) -> np.ndarray:
    """
    Compute absolute pairwise Pearson correlation matrix (training only).

    Returns:
        (D, D) matrix with zeros on diagonal.
    """
    corr = np.abs(np.corrcoef(X_train.T))
    np.fill_diagonal(corr, 0.0)
    return corr


# ── Combined Objective ───────────────────────────────────────────────

class FeatureSelectionObjective:
    """
    Encapsulates the feature-selection objective so it can be evaluated
    for any binary selection vector x ∈ {0,1}^D.

    The objective is fitted exclusively on (X_train, y_train).
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_k: int,
        alpha: float = QAOA_PENALTY_LAMBDA,
        beta: float = QAOA_CARDINALITY_PENALTY,
        importance_weight: float = QAOA_IMPORTANCE_WEIGHT,
        seed: int = SEED,
        relevance_method: str = "mutual_info",
    ):
        self.D = X_train.shape[1]
        self.target_k = target_k
        self.alpha = alpha
        self.beta = beta
        self.importance_weight = importance_weight
        self.seed = seed
        self.relevance_method = relevance_method

        # Compute components on training data
        if relevance_method == "mutual_info":
            self.relevance = compute_mutual_information(X_train, y_train, seed)
        elif relevance_method == "univariate_auc":
            self.relevance = compute_univariate_auc(X_train, y_train, seed)
        else:
            raise ValueError(f"Unknown relevance method: {relevance_method}")

        self.correlation = compute_correlation_matrix(X_train)

        # Normalise relevance to [0, 1] for balanced weighting
        rel_max = self.relevance.max()
        if rel_max > 0:
            self.relevance_normed = self.relevance / rel_max
        else:
            self.relevance_normed = self.relevance.copy()

    # ── Evaluate objective for a binary vector ──

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the objective for a binary selection vector x ∈ {0,1}^D.

        Returns the MAXIMISATION objective value.
        Higher is better.
        """
        assert x.shape == (self.D,), f"Expected shape ({self.D},), got {x.shape}"
        assert set(np.unique(x)).issubset({0, 1}), "x must be binary"

        # Relevance term
        relevance_score = self.importance_weight * np.dot(self.relevance_normed, x)

        # Redundancy penalty
        redundancy = 0.0
        selected = np.where(x == 1)[0]
        for idx_i, i in enumerate(selected):
            for j in selected[idx_i + 1:]:
                redundancy += self.correlation[i, j]
        redundancy_penalty = self.alpha * redundancy

        # Cardinality penalty
        k_diff = np.sum(x) - self.target_k
        cardinality_penalty = self.beta * k_diff ** 2

        return float(relevance_score - redundancy_penalty - cardinality_penalty)

    def evaluate_minimisation(self, x: np.ndarray) -> float:
        """Negated objective for minimisation solvers (QAOA)."""
        return -self.evaluate(x)

    # ── QUBO matrix ──

    def build_qubo(self) -> np.ndarray:
        """
        Build the QUBO matrix Q such that:
            x^T Q x  =  -Objective(x)    (minimisation form)

        The matrix is symmetric.  Diagonal encodes linear terms,
        off-diagonal encodes pairwise interactions.
        """
        D = self.D
        K = self.target_k
        Q = np.zeros((D, D))

        # Diagonal: −relevance + cardinality penalty contribution
        # The quadratic cardinality penalty (Σ x_i − K)² expands to:
        #   Σ_i x_i² + 2 Σ_{i<j} x_i x_j − 2K Σ_i x_i + K²
        # Since x_i ∈ {0,1}, x_i² = x_i, so linear terms go on diagonal.
        for i in range(D):
            Q[i, i] = (
                -self.importance_weight * self.relevance_normed[i]   # relevance (negated for min)
                + self.beta * (1 - 2 * K)                            # cardinality linear
            )

        # Off-diagonal: redundancy + cardinality coupling
        # For a symmetric matrix, x^T Q x counts Q[i,j]+Q[j,i]=2*Q[i,j]
        # per pair, so we store half the desired coupling on each side.
        for i in range(D):
            for j in range(i + 1, D):
                coupling = (
                    self.alpha * self.correlation[i, j]  # redundancy (positive → penalise)
                    + 2 * self.beta                       # cardinality quadratic coupling
                )
                Q[i, j] = coupling / 2.0
                Q[j, i] = coupling / 2.0

        return Q

    def verify_qubo(self, Q: np.ndarray, n_samples: int = 50) -> dict:
        """
        Independently verify QUBO by comparing x^T Q x against
        evaluate_minimisation(x) for random binary vectors.

        The QUBO omits the constant βK² from the cardinality penalty
        expansion, so:  x^T Q x + βK² = evaluate_minimisation(x)
        """
        rng = np.random.RandomState(self.seed)
        constant_offset = self.beta * self.target_k ** 2
        max_abs_error = 0.0
        errors = []
        for _ in range(n_samples):
            x = rng.randint(0, 2, self.D).astype(float)
            qubo_val = float(x @ Q @ x)
            obj_val = self.evaluate_minimisation(x)

            # x^T Q x + βK² should equal evaluate_minimisation(x)
            predicted = qubo_val + constant_offset
            error = abs(predicted - obj_val)
            errors.append(error)
            max_abs_error = max(max_abs_error, error)

        return {
            "max_abs_error": float(max_abs_error),
            "mean_abs_error": float(np.mean(errors)),
            "constant_offset_beta_k2": float(constant_offset),
            "n_samples": n_samples,
            "passed": max_abs_error < 1e-8,
        }

    # ── Exact brute-force solver ──

    def brute_force_optimal(self) -> dict:
        """
        Enumerate all 2^D subsets, evaluate objective, return the best.
        Only feasible for small D (≤ ~20).
        """
        assert self.D <= 20, f"Brute force infeasible for D={self.D}"
        best_val = -np.inf
        best_x = None
        n_total = 2 ** self.D
        all_values = []

        for bits in range(n_total):
            x = np.array([(bits >> i) & 1 for i in range(self.D)], dtype=float)
            val = self.evaluate(x)
            all_values.append(val)
            if val > best_val:
                best_val = val
                best_x = x.copy()

        selected_indices = np.where(best_x == 1)[0].tolist()
        return {
            "best_objective": float(best_val),
            "best_x": best_x.tolist(),
            "selected_indices": selected_indices,
            "n_selected": len(selected_indices),
            "n_evaluated": n_total,
            "all_values_stats": {
                "min": float(np.min(all_values)),
                "max": float(np.max(all_values)),
                "mean": float(np.mean(all_values)),
                "std": float(np.std(all_values)),
            },
        }

    # ── Serialisation ──

    def save_components(self, path: str | None = None) -> str:
        """Save objective components for reproducibility."""
        if path is None:
            path = os.path.join(PART4_ARTIFACTS_DIR, "objective_components.json")
        data = {
            "D": self.D,
            "target_k": self.target_k,
            "alpha": self.alpha,
            "beta": self.beta,
            "importance_weight": self.importance_weight,
            "relevance_method": self.relevance_method,
            "seed": self.seed,
            "relevance_raw": self.relevance.tolist(),
            "relevance_normed": self.relevance_normed.tolist(),
            "correlation": self.correlation.tolist(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
