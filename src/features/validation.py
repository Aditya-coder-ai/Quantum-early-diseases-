"""
Feature quality validation and metadata generation module for Part 3.

Enforces strict gates on extracted feature representations:
- No NaNs or Infs
- Equal feature counts across train/val/test splits
- Non-zero feature variances (detect collapsed dimensions)
- Summary statistics and deterministic feature schema metadata
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class FeatureQualityValidator:
    """
    Validates quality and structure of feature matrices before downstream processing.
    """
    def __init__(self, variance_threshold: float = 1e-6):
        self.variance_threshold = variance_threshold

    def validate_features(
        self,
        X_train: np.ndarray | pd.DataFrame,
        X_val: np.ndarray | pd.DataFrame,
        X_test: np.ndarray | pd.DataFrame,
        method_name: str = "reduction",
        dim: int = 8
    ) -> Dict[str, Any]:
        """
        Perform quality audit across feature splits.
        """
        tr_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        va_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        te_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        errors = []
        warnings = []

        # 1. NaN and Inf check
        for name, arr in [("train", tr_arr), ("val", va_arr), ("test", te_arr)]:
            if np.isnan(arr).any():
                errors.append(f"NaN values found in {name} features.")
            if np.isinf(arr).any():
                errors.append(f"Infinite values found in {name} features.")

        # 2. Shape consistency check
        if tr_arr.shape[1] != dim:
            errors.append(f"Train feature dim {tr_arr.shape[1]} does not match target dim {dim}.")
        if va_arr.shape[1] != dim:
            errors.append(f"Val feature dim {va_arr.shape[1]} does not match target dim {dim}.")
        if te_arr.shape[1] != dim:
            errors.append(f"Test feature dim {te_arr.shape[1]} does not match target dim {dim}.")

        if not (tr_arr.shape[1] == va_arr.shape[1] == te_arr.shape[1]):
            errors.append("Feature dimension mismatch across train/val/test splits.")

        # 3. Variance and collapsed dimension check (on train split)
        variances = np.var(tr_arr, axis=0)
        collapsed_indices = np.where(variances <= self.variance_threshold)[0].tolist()
        if len(collapsed_indices) > 0:
            warnings.append(f"Found {len(collapsed_indices)} collapsed/near-zero variance feature dimensions: {collapsed_indices}")

        is_valid = len(errors) == 0

        metadata = {
            "method": method_name,
            "target_dim": dim,
            "actual_dims": {
                "train": list(tr_arr.shape),
                "val": list(va_arr.shape),
                "test": list(te_arr.shape),
            },
            "is_valid": is_valid,
            "has_nan": bool(np.isnan(tr_arr).any() or np.isnan(va_arr).any() or np.isnan(te_arr).any()),
            "has_inf": bool(np.isinf(tr_arr).any() or np.isinf(va_arr).any() or np.isinf(te_arr).any()),
            "collapsed_dims_count": len(collapsed_indices),
            "collapsed_dim_indices": collapsed_indices,
            "feature_variances": [float(v) for v in variances],
            "feature_means": [float(m) for m in np.mean(tr_arr, axis=0)],
            "feature_mins": [float(m) for m in np.min(tr_arr, axis=0)],
            "feature_maxs": [float(m) for m in np.max(tr_arr, axis=0)],
            "errors": errors,
            "warnings": warnings,
        }
        return metadata

    def export_feature_set(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        output_dir: str,
        prefix: str = "compact"
    ) -> Dict[str, str]:
        """
        Save features to CSV with deterministic column headers and return paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        dim = X_train.shape[1]
        cols = [f"feat_{i}" for i in range(dim)]

        df_train = pd.DataFrame(X_train, columns=cols)
        df_val = pd.DataFrame(X_val, columns=cols)
        df_test = pd.DataFrame(X_test, columns=cols)

        tr_path = os.path.join(output_dir, f"X_train_{prefix}.csv")
        va_path = os.path.join(output_dir, f"X_val_{prefix}.csv")
        te_path = os.path.join(output_dir, f"X_test_{prefix}.csv")

        df_train.to_csv(tr_path, index=False)
        df_val.to_csv(va_path, index=False)
        df_test.to_csv(te_path, index=False)

        return {
            "train": tr_path,
            "val": va_path,
            "test": te_path,
        }
