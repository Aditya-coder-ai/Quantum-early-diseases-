"""
Production-quality, leakage-safe preprocessing pipeline for MIndMatrix.

Core principles:
1. Strict Data Leakage Prevention: Scalers and imputers are fit exclusively on training data.
   Validation and test sets are transformed using parameters learned solely from the train set.
2. Group / Patient Leakage Protection: When patient identifiers are present, group-aware
   splitting ensures no patient appears in multiple splits.
3. Quantum-Readiness: Final transformed arrays are guaranteed finite, non-null,
   numerically scaled, with deterministic feature ordering.
4. Production-Ready Inference: Standalone single-sample preprocessing and re-loadable artifacts.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_METADATA_DIR, MODELS_DIR,
    TARGET_COLUMN, PATIENT_ID_COLUMN, SCALING_METHOD, HANDLE_MISSING_STRATEGY,
    REMOVE_DUPLICATES
)
from src.preprocessing.validator import DataValidator


class MedicalPreprocessingPipeline:
    """
    Leakage-safe preprocessor that fits only on training data
    and reliably transforms validation, test, and live inference samples.
    """

    def __init__(
        self,
        scaling_method: str = SCALING_METHOD,
        impute_strategy: str = HANDLE_MISSING_STRATEGY,
    ):
        self.scaling_method = scaling_method
        self.impute_strategy = impute_strategy
        
        # Instantiate transformers
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")

        self.imputer = SimpleImputer(strategy=impute_strategy)
        
        # Fitted attributes
        self.feature_names_in_: list[str] = []
        self.train_stats_: dict[str, Any] = {}
        self.is_fitted_: bool = False

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: Any = None) -> "MedicalPreprocessingPipeline":
        """
        Fit imputer and scaler strictly on training data.
        """
        if isinstance(X_train, np.ndarray):
            self.feature_names_in_ = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_df = pd.DataFrame(X_train, columns=self.feature_names_in_)
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names_in_ = list(X_train.columns)
            X_df = X_train.copy()
        else:
            raise TypeError("X_train must be a pandas DataFrame or numpy ndarray.")

        # 1. Fit imputer
        self.imputer.fit(X_df)
        X_imputed = self.imputer.transform(X_df)

        # 2. Fit scaler
        self.scaler.fit(X_imputed)

        # 3. Record training statistics for auditing & validation
        self.train_stats_ = {
            "n_samples": int(len(X_df)),
            "n_features": int(X_df.shape[1]),
            "feature_names": self.feature_names_in_,
            "feature_means": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
            "feature_scales": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            "feature_medians": self.imputer.statistics_.tolist(),
        }
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Transform data using parameters learned strictly from training data.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted on training data before transform().")

        if isinstance(X, pd.DataFrame):
            # Ensure correct column ordering matching training data
            missing_cols = [c for c in self.feature_names_in_ if c not in X.columns]
            if missing_cols:
                raise ValueError(f"Incoming data missing expected training columns: {missing_cols}")
            X_ordered = X[self.feature_names_in_].copy()
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names_in_):
                raise ValueError(f"Array feature dimension {X.shape[1]} does not match training feature count {len(self.feature_names_in_)}")
            X_ordered = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

        # 1. Apply training imputer
        X_imp = self.imputer.transform(X_ordered)

        # 2. Apply training scaler
        X_scaled = self.scaler.transform(X_imp)

        # 3. Quantum-readiness verification
        assert not np.isnan(X_scaled).any(), "NaN values detected post-scaling."
        assert not np.isinf(X_scaled).any(), "Infinite values detected post-scaling."

        return pd.DataFrame(X_scaled, columns=self.feature_names_in_)

    def fit_transform(self, X_train: pd.DataFrame | np.ndarray, y_train: Any = None) -> pd.DataFrame:
        """Fit on train and transform train."""
        return self.fit(X_train, y_train).transform(X_train)

    def transform_single_sample(self, sample: dict[str, float] | pd.Series | np.ndarray) -> np.ndarray:
        """
        Preprocess a single incoming medical record for inference.
        Returns 1D array of shape (n_features,).
        """
        if isinstance(sample, dict):
            sample_df = pd.DataFrame([sample])
        elif isinstance(sample, pd.Series):
            sample_df = pd.DataFrame([sample.to_dict()])
        elif isinstance(sample, np.ndarray):
            if sample.ndim == 1:
                sample_df = pd.DataFrame([sample], columns=self.feature_names_in_)
            else:
                sample_df = pd.DataFrame(sample, columns=self.feature_names_in_)
        else:
            raise TypeError("Sample must be dict, Series, or ndarray.")

        scaled_df = self.transform(sample_df)
        return scaled_df.values[0]

    def save(self, filepath: str):
        """Serialize fitted pipeline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[PREPROC] Preprocessing pipeline saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "MedicalPreprocessingPipeline":
        """Load fitted pipeline from disk."""
        import sys
        if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "MedicalPreprocessingPipeline"):
            sys.modules["__main__"].MedicalPreprocessingPipeline = MedicalPreprocessingPipeline
        pipeline = joblib.load(filepath)
        if not isinstance(pipeline, MedicalPreprocessingPipeline):
            raise TypeError("Loaded object is not a MedicalPreprocessingPipeline.")
        return pipeline


def split_data_leakage_safe(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    patient_id_column: str | None = PATIENT_ID_COLUMN,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    return_patient_groups: bool = False,
) -> tuple:
    """
    Perform a leakage-safe split into Train (70%), Validation (15%), and Test (15%).
    
    If patient_id_column is specified, uses GroupShuffleSplit to prevent patient leakage.
    Otherwise, uses Stratified train_test_split.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Ratios must sum to 1.0, got {total_ratio}"

    y = df[target_column].copy()
    feature_cols = [c for c in df.columns if c not in [target_column, patient_id_column] and c is not None]
    X = df[feature_cols].copy()

    val_test_ratio = val_ratio + test_ratio
    rel_test_ratio = test_ratio / val_test_ratio  # 0.50 of the remaining 30%

    patient_groups = {"train": set(), "val": set(), "test": set()}

    if patient_id_column and patient_id_column in df.columns and df[patient_id_column].nunique() < len(df):
        print(f"[PREPROC] Patient ID '{patient_id_column}' detected with repeated records. Performing Group-Level Splitting...")
        groups = df[patient_id_column].values
        
        # 1. Split Train vs (Val + Test) by Group
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_seed)
        train_idx, val_test_idx = next(gss_outer.split(X, y, groups))
        
        X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
        X_temp, y_temp = X.iloc[val_test_idx].reset_index(drop=True), y.iloc[val_test_idx].reset_index(drop=True)
        groups_temp = groups[val_test_idx]

        # 2. Split Val vs Test by Group
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=rel_test_ratio, random_state=random_seed)
        val_sub_idx, test_sub_idx = next(gss_inner.split(X_temp, y_temp, groups_temp))
        
        X_val, y_val = X_temp.iloc[val_sub_idx].reset_index(drop=True), y_temp.iloc[val_sub_idx].reset_index(drop=True)
        X_test, y_test = X_temp.iloc[test_sub_idx].reset_index(drop=True), y_temp.iloc[test_sub_idx].reset_index(drop=True)
        
        # Verify zero patient overlap
        train_patients = set(groups[train_idx])
        val_patients = set(groups_temp[val_sub_idx])
        test_patients = set(groups_temp[test_sub_idx])
        assert len(train_patients & val_patients) == 0, "Patient leakage detected between train and val!"
        assert len(train_patients & test_patients) == 0, "Patient leakage detected between train and test!"
        assert len(val_patients & test_patients) == 0, "Patient leakage detected between val and test!"
        print(f"[PREPROC] Zero patient leakage verified across {len(train_patients)} train, {len(val_patients)} val, and {len(test_patients)} test patients.")

        patient_groups = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients
        }

    else:
        # Standard Stratified Splitting
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=val_test_ratio, random_state=random_seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=rel_test_ratio, random_state=random_seed, stratify=y_temp
        )

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

    print(f"[PREPROC] Stratified Split: Train={len(X_train)} ({len(X_train)/len(df)*100:.1f}%), "
          f"Val={len(X_val)} ({len(X_val)/len(df)*100:.1f}%), "
          f"Test={len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

    if return_patient_groups:
        return X_train, X_val, X_test, y_train, y_val, y_test, patient_groups
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing_pipeline(
    df: pd.DataFrame | None = None,
    save: bool = True
) -> dict[str, Any]:
    """
    Execute the complete end-to-end Part 2 preprocessing pipeline.
    
    1. Validation
    2. Duplicate handling
    3. Leakage-safe splitting
    4. Training-only fitting of transformations
    5. Validation and test transforms
    6. Quantum-readiness verification
    7. Metadata generation and disk serialization
    """
    print("\n" + "=" * 65)
    print("   MINDMATRIX PART 2: DATA PREPROCESSING & DATA PIPELINE")
    print("=" * 65)

    # 1. Ingest Data
    if df is None:
        raw_csv = os.path.join(DATA_RAW_DIR, "breast_cancer_raw.csv")
        if not os.path.exists(raw_csv):
            from src.data.loader import load_dataset
            load_dataset(save_raw=True)
        df = pd.read_csv(raw_csv)
        print(f"[PREPROC] Loaded raw dataset from {raw_csv} ({df.shape[0]} rows, {df.shape[1]} cols)")

    # 2. Comprehensive Validation Audit
    validator = DataValidator(target_column=TARGET_COLUMN, patient_id_column=PATIENT_ID_COLUMN)
    validation_report = validator.validate(df)
    validator.print_summary(validation_report)

    if not validation_report["is_valid"]:
        raise ValueError(f"Dataset validation failed: {validation_report['errors']}")

    # 3. Clean exact duplicates if configured
    if REMOVE_DUPLICATES and validation_report["duplicate_records"] > 0:
        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[PREPROC] Removed {n_before - len(df)} duplicate records.")

    # 4. Leakage-Safe Splitting
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data_leakage_safe(
        df=df,
        target_column=TARGET_COLUMN,
        patient_id_column=PATIENT_ID_COLUMN,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )

    # 5. Fit Preprocessor ONLY on Training Data
    pipeline = MedicalPreprocessingPipeline(scaling_method=SCALING_METHOD)
    pipeline.fit(X_train_raw, y_train)

    # 6. Transform All Splits
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # 7. Quantum-Readiness Assertions
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Dimension mismatch across splits."
    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns), "Feature ordering mismatch across splits."
    assert not X_train.isnull().any().any(), "NaNs found in X_train."
    assert not X_val.isnull().any().any(), "NaNs found in X_val."
    assert not X_test.isnull().any().any(), "NaNs found in X_test."

    # 8. Compile Metadata
    feature_schema = {
        "feature_count": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "feature_types": {col: str(X_train[col].dtype) for col in X_train.columns},
        "target_column": TARGET_COLUMN,
        "target_classes": {
            "0": "Malignant",
            "1": "Benign"
        }
    }

    preprocessing_meta = {
        "random_seed": RANDOM_SEED,
        "scaling_method": SCALING_METHOD,
        "handle_missing_strategy": HANDLE_MISSING_STRATEGY,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO
        },
        "split_sample_counts": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
            "total": int(len(df))
        },
        "class_distributions": {
            "train": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "val": {str(k): int(v) for k, v in y_val.value_counts().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().items()}
        },
        "patient_id_column": PATIENT_ID_COLUMN,
        "patient_leakage_detected": False,
        "train_fitted_means": pipeline.train_stats_["feature_means"],
        "train_fitted_scales": pipeline.train_stats_["feature_scales"],
    }

    # 9. Save Processed Artifacts & Metadata
    if save:
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        os.makedirs(DATA_METADATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        # CSVs
        X_train.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"), index=False)
        X_val.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"), index=False)
        X_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv"), index=False)
        y_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv"), index=False)

        # Preprocessing artifacts
        pipeline.save(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))
        joblib.dump(pipeline.scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

        # Metadata files in data/metadata/
        with open(os.path.join(DATA_METADATA_DIR, "dataset_report.json"), "w") as f:
            json.dump(validation_report, f, indent=2)

        with open(os.path.join(DATA_METADATA_DIR, "feature_schema.json"), "w") as f:
            json.dump(feature_schema, f, indent=2)

        with open(os.path.join(DATA_METADATA_DIR, "preprocessing_metadata.json"), "w") as f:
            json.dump(preprocessing_meta, f, indent=2)

        # Backwards compatibility file for downstream modules
        split_meta_legacy = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "train_class_dist": {int(k): int(v) for k, v in y_train.value_counts().items()},
            "val_class_dist": {int(k): int(v) for k, v in y_val.value_counts().items()},
            "test_class_dist": {int(k): int(v) for k, v in y_test.value_counts().items()},
        }
        with open(os.path.join(DATA_PROCESSED_DIR, "split_metadata.json"), "w") as f:
            json.dump(split_meta_legacy, f, indent=2)

        # Markdown report
        md_report = f"""# Preprocessing Audit Report & Data Pipeline Summary

## 1. Dataset Overview
- **Raw Samples:** {len(df)}
- **Feature Count:** {X_train.shape[1]}
- **Target Column:** {TARGET_COLUMN} (0=Malignant, 1=Benign)

## 2. Leakage-Safe Splits
- **Train:** {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%) | Classes: {dict(y_train.value_counts())}
- **Validation:** {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%) | Classes: {dict(y_val.value_counts())}
- **Test:** {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%) | Classes: {dict(y_test.value_counts())}

## 3. Transformation & Scaling
- **Scaler Method:** {SCALING_METHOD}
- **Fitted Strictly On:** Training Set ($N={len(X_train)}$)
- **Validation/Test Sets:** Transformed via Training-learned parameters (Zero Leakage)
- **Quantum Readiness:** Checked (No NaNs, No Infs, Float64 normalized values)
"""
        with open(os.path.join(DATA_METADATA_DIR, "dataset_report.md"), "w", encoding="utf-8") as f:
            f.write(md_report)

        print(f"[PREPROC] Processed CSV files saved to {DATA_PROCESSED_DIR}")
        print(f"[PREPROC] Metadata files saved to {DATA_METADATA_DIR}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pipeline": pipeline,
        "validation_report": validation_report,
        "feature_schema": feature_schema,
        "preprocessing_metadata": preprocessing_meta,
    }


if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    print("\n[PREPROC] Pipeline successfully executed.")
