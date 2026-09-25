"""
Dataset validation module for MIndMatrix Medical Disease Detection System.

Implements rigorous checks:
- Dataset existence and loadability
- Target column presence and binary validity
- Missing values and infinite values detection
- Duplicate records detection
- Constant / near-zero variance feature detection
- Data types and unexpected categorical values
- Patient / Subject ID identification & isolation
- Class imbalance measurement
- Outlier diagnostics
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Any


class DataValidator:
    """
    Comprehensive validator for medical tabular datasets.
    Ensures data integrity, detects anomalies, and prevents leakage.
    """
    def __init__(
        self,
        target_column: str = "target",
        patient_id_column: str | None = None,
        variance_threshold: float = 1e-7,
    ):
        self.target_column = target_column
        self.patient_id_column = patient_id_column
        self.variance_threshold = variance_threshold

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run all validation checks on the dataset.
        
        Returns:
            Dictionary containing validation results, errors, warnings, and statistics.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Existence and non-emptiness
        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "is_valid": False,
                "errors": ["Input data is not a valid pandas DataFrame."],
                "warnings": [],
            }
        
        n_samples, n_total_cols = df.shape
        if n_samples == 0:
            errors.append("Dataset contains 0 rows/samples.")
        if n_total_cols == 0:
            errors.append("Dataset contains 0 columns.")

        # 2. Target Column Check
        target_present = self.target_column in df.columns
        target_stats = {}
        if not target_present:
            errors.append(f"Target column '{self.target_column}' not found in dataset columns.")
        else:
            target_series = df[self.target_column]
            target_nulls = int(target_series.isnull().sum())
            if target_nulls > 0:
                errors.append(f"Target column contains {target_nulls} missing/null values.")
            
            unique_targets = target_series.dropna().unique()
            n_classes = len(unique_targets)
            if n_classes != 2:
                errors.append(f"Expected binary classification target (2 classes), found {n_classes} classes: {unique_targets.tolist()}")

            class_counts = target_series.value_counts().to_dict()
            class_percentages = (target_series.value_counts(normalize=True) * 100).round(2).to_dict()
            min_count = min(class_counts.values()) if class_counts else 0
            max_count = max(class_counts.values()) if class_counts else 0
            imbalance_ratio = round(max_count / min_count, 3) if min_count > 0 else float("inf")

            target_stats = {
                "classes": [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in unique_targets],
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "class_percentages": {str(k): float(v) for k, v in class_percentages.items()},
                "imbalance_ratio": imbalance_ratio,
            }
            if imbalance_ratio > 3.0:
                warnings.append(f"High class imbalance detected (ratio {imbalance_ratio}:1).")

        # 3. Patient ID Identification & Isolation
        patient_stats = {}
        if self.patient_id_column:
            if self.patient_id_column not in df.columns:
                warnings.append(f"Specified patient ID column '{self.patient_id_column}' not found in dataframe.")
            else:
                n_patients = int(df[self.patient_id_column].nunique())
                repeated_patients = int(n_samples - n_patients)
                patient_stats = {
                    "patient_id_column": self.patient_id_column,
                    "unique_patients": n_patients,
                    "repeated_measurements": repeated_patients,
                    "requires_grouped_splitting": repeated_patients > 0,
                }
                if repeated_patients > 0:
                    warnings.append(f"Dataset contains repeated patient measurements ({repeated_patients} repeats). Grouped splitting required.")
        else:
            patient_stats = {
                "patient_id_column": None,
                "note": "No patient ID column specified; each row treated as independent biopsy instance."
            }

        # 4. Feature Columns Identification
        cols_to_exclude = [self.target_column]
        if self.patient_id_column and self.patient_id_column in df.columns:
            cols_to_exclude.append(self.patient_id_column)
        
        feature_cols = [c for c in df.columns if c not in cols_to_exclude]
        X = df[feature_cols]

        # 5. Missing Values Check
        missing_series = X.isnull().sum()
        total_missing = int(missing_series.sum())
        cols_with_missing = {col: int(cnt) for col, cnt in missing_series.items() if cnt > 0}
        if total_missing > 0:
            warnings.append(f"Dataset contains {total_missing} missing values across {len(cols_with_missing)} feature columns.")

        # 6. Duplicate Rows Check (Feature space)
        exact_duplicates = int(X.duplicated().sum())
        if exact_duplicates > 0:
            warnings.append(f"Dataset contains {exact_duplicates} duplicate feature records.")

        # 7. Infinite / Invalid Numeric Values
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in feature_cols if c not in numeric_cols]
        inf_counts = {}
        for col in numeric_cols:
            inf_cnt = int(np.isinf(X[col]).sum())
            if inf_cnt > 0:
                inf_counts[col] = inf_cnt
        total_inf = sum(inf_counts.values())
        if total_inf > 0:
            errors.append(f"Found {total_inf} infinite values across {len(inf_counts)} numeric columns: {inf_counts}")

        # 8. Constant / Zero-Variance Features
        constant_features = []
        for col in numeric_cols:
            var_val = float(X[col].var())
            if var_val <= self.variance_threshold:
                constant_features.append({"column": col, "variance": var_val})
        if constant_features:
            warnings.append(f"Found {len(constant_features)} constant/near-zero variance features: {[c['column'] for c in constant_features]}")

        # 9. Outlier Diagnostics (IQR Method - non-destructive audit)
        outlier_summary = {}
        for col in numeric_cols:
            q25 = float(X[col].quantile(0.25))
            q75 = float(X[col].quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            n_outliers = int(((X[col] < lower_bound) | (X[col] > upper_bound)).sum())
            outlier_summary[col] = {
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / n_samples * 100, 2),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4)
            }

        is_valid = len(errors) == 0

        report = {
            "is_valid": is_valid,
            "n_samples": n_samples,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": non_numeric_cols,
            "target_statistics": target_stats,
            "patient_statistics": patient_stats,
            "missing_values": {
                "total_missing": total_missing,
                "columns_with_missing": cols_with_missing,
            },
            "duplicate_records": exact_duplicates,
            "infinite_values": inf_counts,
            "constant_features": constant_features,
            "outlier_summary": outlier_summary,
            "errors": errors,
            "warnings": warnings,
        }
        return report

    def print_summary(self, report: dict[str, Any]):
        """Print a human-readable validation summary."""
        print("\n" + "=" * 60)
        print("DATASET VALIDATION AUDIT REPORT")
        print("=" * 60)
        print(f"Validation Status: {'PASSED [OK]' if report['is_valid'] else 'FAILED [ERRORS DETECTED]'}")
        print(f"Total Samples:     {report.get('n_samples', 0)}")
        print(f"Total Features:    {report.get('n_features', 0)}")
        print(f"Missing Values:    {report.get('missing_values', {}).get('total_missing', 0)}")
        print(f"Duplicate Rows:    {report.get('duplicate_records', 0)}")
        print(f"Infinite Values:   {sum(report.get('infinite_values', {}).values())}")
        print(f"Zero-Var Features: {len(report.get('constant_features', []))}")
        
        target = report.get("target_statistics", {})
        if target:
            print(f"Class Counts:      {target.get('class_counts')}")
            print(f"Imbalance Ratio:   {target.get('imbalance_ratio')}:1")
        
        if report.get("errors"):
            print("\n[CRITICAL ERRORS]")
            for err in report["errors"]:
                print(f"  ❌ {err}")
        
        if report.get("warnings"):
            print("\n[AUDIT WARNINGS]")
            for warn in report["warnings"]:
                print(f"  ⚠️ {warn}")
        print("=" * 60)
