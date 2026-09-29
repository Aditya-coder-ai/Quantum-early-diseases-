"""
Part 6: Feature Normalization & Quantum Encoding.
Provides zero-leakage AngleScaler and deterministic feature-to-qubit mapping.
"""
from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class AngleScaler:
    """
    Fits feature bounds strictly on training data and transforms continuous features
    into bounded angle representations in [0, pi] for quantum rotation gates (e.g., RY).
    Guarantees zero data leakage from validation or test splits.
    """
    def __init__(self, target_range: Tuple[float, float] = (0.0, np.pi)):
        self.target_min, self.target_max = target_range
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.range_vals: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> AngleScaler:
        """
        Compute minimum and maximum bounds strictly from training features.
        
        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
        """
        X_arr = np.asarray(X_train, dtype=np.float64)
        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValueError("Training data contains NaN or infinite values.")

        self.min_vals = np.min(X_arr, axis=0)
        self.max_vals = np.max(X_arr, axis=0)
        self.range_vals = self.max_vals - self.min_vals
        
        # Prevent division by zero for invariant features
        self.range_vals[self.range_vals < 1e-10] = 1.0
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] using fitted training parameters and scale to [target_min, target_max].
        Clips out-of-bound values to maintain strict quantum angle constraints.
        """
        if not self.is_fitted or self.min_vals is None or self.range_vals is None:
            raise RuntimeError("AngleScaler must be fitted on training data before transform.")

        X_arr = np.asarray(X, dtype=np.float64)
        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValueError("Input data contains NaN or infinite values.")

        # Zero-leakage scaling with fitted bounds
        X_norm = (X_arr - self.min_vals) / self.range_vals
        X_clipped = np.clip(X_norm, 0.0, 1.0)
        
        # Scale to target angle range [target_min, target_max]
        angles = self.target_min + X_clipped * (self.target_max - self.target_min)
        return angles.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and return transformed angles."""
        return self.fit(X_train).transform(X_train)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler bounds for reproducibility and artifact persistence."""
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet.")
        return {
            "is_fitted": self.is_fitted,
            "min_vals": self.min_vals.tolist() if self.min_vals is not None else None,
            "max_vals": self.max_vals.tolist() if self.max_vals is not None else None,
            "range_vals": self.range_vals.tolist() if self.range_vals is not None else None,
            "target_min": float(self.target_min),
            "target_max": float(self.target_max),
        }

    def save(self, filepath: str) -> None:
        """Save fitted bounds to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> AngleScaler:
        """Load fitted AngleScaler from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        scaler = cls(target_range=(data["target_min"], data["target_max"]))
        scaler.min_vals = np.array(data["min_vals"], dtype=np.float64)
        scaler.max_vals = np.array(data["max_vals"], dtype=np.float64)
        scaler.range_vals = np.array(data["range_vals"], dtype=np.float64)
        scaler.is_fitted = data["is_fitted"]
        return scaler


class FeatureQubitMapper:
    """
    Manages deterministic 1-to-1 mapping between continuous compact feature columns
    and quantum register wires.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)
        self.n_features = len(self.feature_names)
        self.mapping = {idx: name for idx, name in enumerate(self.feature_names)}
        self.qubit_to_feature = {qubit: idx for qubit, idx in enumerate(range(self.n_features))}

    def get_qubit_for_feature(self, feature_idx: int) -> int:
        """Return target qubit wire index for given feature index."""
        if feature_idx < 0 or feature_idx >= self.n_features:
            raise IndexError(f"Feature index {feature_idx} out of range [0, {self.n_features - 1}].")
        return feature_idx

    def get_feature_name(self, qubit_wire: int) -> str:
        """Return feature name mapped to qubit wire."""
        return self.feature_names[qubit_wire]

    def slice_features(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, List[str]]:
        """
        Extract the first k features according to the configured feature priority.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            k: Desired feature count / qubit count
        Returns:
            (X_k, feature_names_k)
        """
        if k > self.n_features:
            raise ValueError(f"Requested k={k} exceeds available features {self.n_features}.")
        X_sliced = X[:, :k]
        names_sliced = self.feature_names[:k]
        return X_sliced, names_sliced
