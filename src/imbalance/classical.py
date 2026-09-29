"""
Part 5 — Classical Class Imbalance Strategies.

Implements:
1. Exact imbalance measurement on training data
2. Class weighting calculation
3. Synthetic Minority Over-sampling Technique (SMOTE) from first principles
4. Zero-leakage verification guarantees
"""
from __future__ import annotations

import json
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.neighbors import NearestNeighbors

from src.imbalance.config import (
    MINORITY_CLASS, MAJORITY_CLASS, SMOTE_K_NEIGHBORS, SEED, PART5_ARTIFACTS_DIR
)

logger = logging.getLogger(__name__)


# ── STEP 1: Imbalance Measurement ─────────────────────────────────────

def measure_imbalance(
    y_train: np.ndarray,
    minority_class: int = MINORITY_CLASS,
    majority_class: int = MAJORITY_CLASS,
) -> Dict[str, Any]:
    """
    Measure the class imbalance distribution on the training data.

    Returns:
        Dict with sample counts, percentages, minority ratio, and imbalance ratio.
    """
    y_arr = np.asarray(y_train).ravel()
    n_total = len(y_arr)
    classes, counts = np.unique(y_arr, return_counts=True)
    count_dict = {int(c): int(cnt) for c, cnt in zip(classes, counts)}

    n_minority = count_dict.get(minority_class, 0)
    n_majority = count_dict.get(majority_class, 0)

    if n_minority == 0 or n_majority == 0:
        raise ValueError(f"Both classes must exist in y_train. Found counts: {count_dict}")

    minority_pct = round((n_minority / n_total) * 100, 2)
    majority_pct = round((n_majority / n_total) * 100, 2)
    minority_ratio = round(n_minority / n_majority, 4)
    imbalance_ratio = round(n_majority / n_minority, 4)
    deficit = n_majority - n_minority

    summary = {
        "total_samples": n_total,
        "class_counts": count_dict,
        "minority_class": minority_class,
        "minority_samples": n_minority,
        "minority_percentage": minority_pct,
        "majority_class": majority_class,
        "majority_samples": n_majority,
        "majority_percentage": majority_pct,
        "minority_to_majority_ratio": minority_ratio,
        "imbalance_ratio": imbalance_ratio,
        "deficit": deficit,
    }
    return summary


# ── STEP 3: Class Weights ─────────────────────────────────────────────

def compute_class_weights(
    y_train: np.ndarray,
    minority_class: int = MINORITY_CLASS,
    majority_class: int = MAJORITY_CLASS,
) -> Dict[int, float]:
    """
    Compute balanced class weights for cost-sensitive learning.
    w_c = N / (2 * N_c)
    """
    info = measure_imbalance(y_train, minority_class, majority_class)
    n_total = info["total_samples"]
    n_min = info["minority_samples"]
    n_maj = info["majority_samples"]

    w_min = round(float(n_total / (2.0 * n_min)), 4)
    w_maj = round(float(n_total / (2.0 * n_maj)), 4)

    return {minority_class: w_min, majority_class: w_maj}


# ── STEP 4: SMOTE Implementation ──────────────────────────────────────

def smote_oversample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ratio: float = 1.0,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    seed: int = SEED,
    minority_class: int = MINORITY_CLASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) implemented from first principles.

    CRITICAL LEAKAGE RULE:
    Fitted and applied strictly to (X_train, y_train). Never sees validation or test data.

    Parameters:
        X_train: Training features (N, D)
        y_train: Training labels (N,)
        ratio: Fraction of the minority deficit to generate.
               1.0 = generate enough to match majority count (50/50 balance)
               0.5 = generate 50% of the deficit
        k_neighbors: Number of nearest neighbors to consider in minority class
        seed: Random seed for reproducibility
        minority_class: Label of the minority class

    Returns:
        X_augmented: Combined [X_train, X_synthetic]
        y_augmented: Combined [y_train, y_synthetic]
        X_synthetic: Only the generated synthetic samples
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train).ravel()

    # Isolate minority samples
    min_indices = np.where(y_tr == minority_class)[0]
    X_min = X_tr[min_indices]
    n_min = len(X_min)
    n_maj = len(y_tr) - n_min
    deficit = max(0, n_maj - n_min)

    n_to_generate = int(round(deficit * ratio))

    if n_to_generate <= 0 or n_min < 2:
        logger.warning("No synthetic samples requested or needed (ratio=%f, deficit=%d)", ratio, deficit)
        empty_synth = np.empty((0, X_tr.shape[1]), dtype=np.float64)
        return X_tr.copy(), y_tr.copy(), empty_synth

    rng = np.random.RandomState(seed)

    # Fit k-nearest neighbors on the minority class only
    k_eff = min(k_neighbors, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X_min)
    _, indices = nn.kneighbors(X_min)  # shape (n_min, k_eff + 1)
    # Exclude self (index 0)
    neighbor_indices = indices[:, 1:]

    # Generate synthetic samples
    synthetic_samples = np.zeros((n_to_generate, X_tr.shape[1]), dtype=np.float64)
    for i in range(n_to_generate):
        # Pick a random minority sample
        sample_idx = rng.randint(0, n_min)
        base_sample = X_min[sample_idx]

        # Pick one of its k nearest neighbors
        neighbor_col = rng.randint(0, k_eff)
        chosen_neighbor_idx = neighbor_indices[sample_idx, neighbor_col]
        neighbor_sample = X_min[chosen_neighbor_idx]

        # Interpolate along line segment
        lambda_val = rng.uniform(0.0, 1.0)
        synthetic_samples[i] = base_sample + lambda_val * (neighbor_sample - base_sample)

    assert not np.isnan(synthetic_samples).any(), "SMOTE generated NaN values"
    assert not np.isinf(synthetic_samples).any(), "SMOTE generated infinite values"
    assert synthetic_samples.shape == (n_to_generate, X_tr.shape[1])

    # Combine into augmented training dataset
    X_augmented = np.vstack([X_tr, synthetic_samples])
    y_augmented = np.hstack([y_tr, np.full(n_to_generate, minority_class, dtype=y_tr.dtype)])

    return X_augmented, y_augmented, synthetic_samples
