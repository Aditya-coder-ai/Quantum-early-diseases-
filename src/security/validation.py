"""
Input validation for the inference/API layer.

Validates incoming data for prediction requests:
- Required fields
- Data types (numeric)
- Numeric ranges (physiologically plausible)
- Feature count matches trained model
- No unexpected columns
- Missing value handling
- Payload size limits
- No internal stack traces in error responses

Returns safe, structured error messages only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# ── WDBC Feature Ranges (min/max from UCI dataset with generous margins) ──
# These are physiologically plausible bounds for the 30 WDBC features.
# Inputs outside these bounds are flagged as suspicious but not always rejected
# (the model may still produce a result; the warning is informational).
WDBC_FEATURE_COUNT = 30

WDBC_FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity", "mean concave points",
    "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error", "concave points error",
    "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity", "worst concave points",
    "worst symmetry", "worst fractal dimension",
]

# Generous min/max bounds (0 and 5× observed max in UCI dataset)
_FEATURE_BOUNDS = {
    "mean radius": (0, 150),
    "mean texture": (0, 200),
    "mean perimeter": (0, 1100),
    "mean area": (0, 13000),
    "mean smoothness": (0, 1),
    "mean compactness": (0, 2),
    "mean concavity": (0, 2),
    "mean concave points": (0, 1),
    "mean symmetry": (0, 1),
    "mean fractal dimension": (0, 1),
    "radius error": (0, 15),
    "texture error": (0, 25),
    "perimeter error": (0, 110),
    "area error": (0, 2700),
    "smoothness error": (0, 0.2),
    "compactness error": (0, 0.6),
    "concavity error": (0, 0.6),
    "concave points error": (0, 0.2),
    "symmetry error": (0, 0.5),
    "fractal dimension error": (0, 0.2),
    "worst radius": (0, 200),
    "worst texture": (0, 250),
    "worst perimeter": (0, 1500),
    "worst area": (0, 25000),
    "worst smoothness": (0, 1),
    "worst compactness": (0, 6),
    "worst concavity": (0, 6),
    "worst concave points": (0, 1),
    "worst symmetry": (0, 2),
    "worst fractal dimension": (0, 1),
}

# Maximum payload size (number of samples in a single request)
MAX_BATCH_SIZE = 1000


@dataclass
class ValidationResult:
    """Structured result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def to_safe_response(self) -> Dict[str, Any]:
        """
        Return a safe error response (no stack traces, no internal paths).
        """
        if self.is_valid:
            return {"valid": True}
        return {
            "valid": False,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_prediction_input(
    data: Any,
    expected_features: Optional[List[str]] = None,
    expected_feature_count: Optional[int] = None,
) -> ValidationResult:
    """
    Validate input data for a prediction request.

    Args:
        data: Raw input — can be a dict (single sample), list of dicts,
              DataFrame, or numpy array.
        expected_features: Expected feature column names (from trained model).
        expected_feature_count: Expected number of features.

    Returns:
        ValidationResult with is_valid flag, errors, and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if expected_features is None:
        expected_features = WDBC_FEATURE_NAMES
    if expected_feature_count is None:
        expected_feature_count = len(expected_features)

    # 1. Type check
    if data is None:
        errors.append("Input data is null/None.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 2. Convert to DataFrame
    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            if len(data) == 0:
                errors.append("Input data is an empty list.")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            errors.append(f"Unsupported input type: {type(data).__name__}. Expected dict, list, DataFrame, or ndarray.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    except Exception:
        errors.append("Failed to parse input data into a tabular format.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 3. Batch size limit
    if len(df) > MAX_BATCH_SIZE:
        errors.append(f"Batch size {len(df)} exceeds maximum allowed ({MAX_BATCH_SIZE}).")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    if len(df) == 0:
        errors.append("Input data contains zero samples.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 4. Feature count
    if df.shape[1] != expected_feature_count:
        errors.append(
            f"Feature count mismatch: got {df.shape[1]}, expected {expected_feature_count}."
        )

    # 5. Required features (if column names are available)
    if list(df.columns) != list(range(df.shape[1])):  # named columns
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            errors.append(f"Missing required features: {missing[:5]}{'...' if len(missing) > 5 else ''}.")

        unexpected = [c for c in df.columns if c not in expected_features]
        if unexpected:
            warnings.append(f"Unexpected columns will be ignored: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}.")

    # 6. Missing values
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        null_cols = [str(c) for c in df.columns[df.isnull().any()]]
        errors.append(f"Input contains {null_count} missing value(s) in columns: {null_cols[:5]}.")

    # 7. Numeric type check
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Feature '{col}' is not numeric (got {df[col].dtype}).")

    # 8. Infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_df.values).sum()) if len(numeric_df.columns) > 0 else 0
    if inf_count > 0:
        errors.append(f"Input contains {inf_count} infinite value(s).")

    # 9. Range checks (warnings, not hard errors)
    if list(df.columns) != list(range(df.shape[1])):  # named columns
        for col in df.columns:
            if col in _FEATURE_BOUNDS:
                lo, hi = _FEATURE_BOUNDS[col]
                col_vals = pd.to_numeric(df[col], errors="coerce")
                if col_vals.min() < lo or col_vals.max() > hi:
                    warnings.append(
                        f"Feature '{col}' has values outside expected range [{lo}, {hi}]."
                    )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def safe_error_response(message: str, status_code: int = 400) -> Dict[str, Any]:
    """
    Create a safe error response with no internal details.
    """
    return {
        "error": message,
        "status": status_code,
    }
