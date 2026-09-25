"""
Comprehensive automated tests for Part 2: Data Preprocessing & Data Pipeline.

Tests all 12 mandatory criteria:
1. Dataset loads successfully.
2. Required target exists.
3. Output contains no NaN values.
4. Output contains no infinite values.
5. Train/validation/test dimensions are correct.
6. Class labels are valid.
7. Feature ordering is identical across splits.
8. Patient leakage does not exist when patient IDs are available.
9. Test data was not used to fit preprocessing.
10. Running with the same seed gives reproducible results.
11. Preprocessing a single new sample produces the expected feature dimension.
12. Saved preprocessing objects can be reloaded successfully.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from configs.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_METADATA_DIR, MODELS_DIR,
    TARGET_COLUMN, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from src.preprocessing.validator import DataValidator
from src.preprocessing.pipeline import (
    MedicalPreprocessingPipeline,
    split_data_leakage_safe,
    run_preprocessing_pipeline
)


class TestPreprocessingPipeline:
    """Comprehensive test suite for Part 2 Data Preprocessing."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure preprocessed data exists before tests."""
        self.raw_path = os.path.join(DATA_RAW_DIR, "breast_cancer_raw.csv")
        assert os.path.exists(self.raw_path), f"Raw data not found at {self.raw_path}"
        self.df_raw = pd.read_csv(self.raw_path)

    def test_01_dataset_loads_successfully(self):
        """1. Verify dataset loads and contains expected shape."""
        assert isinstance(self.df_raw, pd.DataFrame)
        assert len(self.df_raw) == 569, f"Expected 569 rows, got {len(self.df_raw)}"
        assert self.df_raw.shape[1] == 31, f"Expected 31 columns (30 features + target), got {self.df_raw.shape[1]}"

    def test_02_required_target_exists(self):
        """2. Verify required target column exists with valid binary labels."""
        assert TARGET_COLUMN in self.df_raw.columns, f"Target column '{TARGET_COLUMN}' missing."
        y = self.df_raw[TARGET_COLUMN]
        assert not y.isnull().any(), "Target contains null values."
        unique_targets = set(y.unique())
        assert unique_targets == {0, 1}, f"Target labels must be {{0, 1}}, got {unique_targets}"

    def test_03_output_contains_no_nan(self):
        """3. Verify processed train, val, and test matrices contain no NaNs."""
        for split in ["train", "val", "test"]:
            x_file = os.path.join(DATA_PROCESSED_DIR, f"X_{split}.csv")
            y_file = os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")
            assert os.path.exists(x_file), f"Missing {x_file}"
            assert os.path.exists(y_file), f"Missing {y_file}"
            
            X = pd.read_csv(x_file)
            y = pd.read_csv(y_file)
            assert not X.isnull().any().any(), f"NaNs found in processed X_{split}."
            assert not y.isnull().any().any(), f"NaNs found in processed y_{split}."

    def test_04_output_contains_no_inf(self):
        """4. Verify processed train, val, and test matrices contain no infinite values."""
        for split in ["train", "val", "test"]:
            X = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, f"X_{split}.csv"))
            numeric_vals = X.select_dtypes(include=[np.number]).values
            assert not np.isinf(numeric_vals).any(), f"Infinite values found in X_{split}."

    def test_05_train_val_test_dimensions(self):
        """5. Verify train/val/test splits match expected sample and feature dimensions."""
        X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"))
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))
        
        y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv"))
        y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv"))
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv"))

        # Dimensions: 398 train (70%), 85 val (15%), 86 test (15%)
        assert len(X_train) == 398, f"Expected 398 train samples, got {len(X_train)}"
        assert len(X_val) == 85, f"Expected 85 val samples, got {len(X_val)}"
        assert len(X_test) == 86, f"Expected 86 test samples, got {len(X_test)}"
        assert len(X_train) + len(X_val) + len(X_test) == 569, "Total samples do not sum to 569."

        # Target dimensions match feature dimensions
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Feature count: 30 continuous features
        assert X_train.shape[1] == 30
        assert X_val.shape[1] == 30
        assert X_test.shape[1] == 30

    def test_06_class_labels_are_valid(self):
        """6. Verify all class labels across splits are strictly binary {0, 1}."""
        for split in ["train", "val", "test"]:
            y = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")).squeeze()
            classes = set(y.unique())
            assert classes.issubset({0, 1}), f"Invalid classes in y_{split}: {classes}"

    def test_07_feature_ordering_identical(self):
        """7. Verify feature ordering and names are identical across all splits."""
        X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"))
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))

        train_cols = list(X_train.columns)
        val_cols = list(X_val.columns)
        test_cols = list(X_test.columns)

        assert train_cols == val_cols, "Feature ordering mismatch between Train and Val."
        assert train_cols == test_cols, "Feature ordering mismatch between Train and Test."

    def test_08_patient_leakage_does_not_exist(self):
        """8. Verify patient/group-level leakage safety."""
        # A: Verify current dataset has no patient column causing leakage
        report_path = os.path.join(DATA_METADATA_DIR, "preprocessing_metadata.json")
        with open(report_path) as f:
            meta = json.load(f)
        assert meta["patient_leakage_detected"] is False

        # B: Stress-test group splitting logic with synthetic multi-measurement patient data
        np.random.seed(RANDOM_SEED)
        n_samples = 120
        n_patients = 20  # Each patient has ~6 records
        synthetic_patients = np.repeat(np.arange(n_patients), n_samples // n_patients)
        synthetic_df = pd.DataFrame(
            np.random.randn(n_samples, 5),
            columns=[f"feat_{i}" for i in range(5)]
        )
        synthetic_df["patient_id"] = synthetic_patients
        synthetic_df["target"] = np.random.randint(0, 2, n_samples)

        X_tr, X_v, X_te, y_tr, y_v, y_te, pt_groups = split_data_leakage_safe(
            df=synthetic_df,
            target_column="target",
            patient_id_column="patient_id",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=RANDOM_SEED,
            return_patient_groups=True
        )

        train_pts = pt_groups["train"]
        val_pts = pt_groups["val"]
        test_pts = pt_groups["test"]

        assert len(train_pts & val_pts) == 0, "Patient leakage detected between train and val!"
        assert len(train_pts & test_pts) == 0, "Patient leakage detected between train and test!"
        assert len(val_pts & test_pts) == 0, "Patient leakage detected between val and test!"

    def test_09_test_data_not_used_to_fit_preprocessing(self):
        """9. Mathematically verify scaler was fitted strictly on training data."""
        # Load raw splits
        X_train_raw, X_val_raw, X_test_raw, _, _, _ = split_data_leakage_safe(
            df=self.df_raw,
            target_column=TARGET_COLUMN,
            random_seed=RANDOM_SEED
        )

        pipeline = MedicalPreprocessingPipeline.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))
        scaler = pipeline.scaler

        # Scaler mean must exactly match X_train_raw mean
        expected_train_mean = X_train_raw.mean().values
        actual_scaler_mean = scaler.mean_
        assert np.allclose(actual_scaler_mean, expected_train_mean, atol=1e-5), (
            "Scaler mean does NOT match training set mean. Possible data leakage!"
        )

        # Scaler mean must NOT equal combined train+test mean (proves test was excluded)
        combined_raw = pd.concat([X_train_raw, X_test_raw], axis=0)
        combined_mean = combined_raw.mean().values
        assert not np.allclose(actual_scaler_mean, combined_mean, atol=1e-4), (
            "Scaler mean matches combined train+test data. Test data was leaked into scaler!"
        )

    def test_10_reproducibility_with_same_seed(self):
        """10. Verify executing preprocessing with the same seed produces identical results."""
        res1 = run_preprocessing_pipeline(df=self.df_raw, save=False)
        res2 = run_preprocessing_pipeline(df=self.df_raw, save=False)

        np.testing.assert_array_equal(res1["X_train"].values, res2["X_train"].values)
        np.testing.assert_array_equal(res1["X_val"].values, res2["X_val"].values)
        np.testing.assert_array_equal(res1["X_test"].values, res2["X_test"].values)
        np.testing.assert_array_equal(res1["y_train"].values, res2["y_train"].values)

    def test_11_single_sample_preprocessing(self):
        """11. Verify preprocessing a single new sample produces expected dimension."""
        pipeline = MedicalPreprocessingPipeline.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))
        
        # Take a raw single row as dictionary
        feature_cols = [c for c in self.df_raw.columns if c != TARGET_COLUMN]
        sample_dict = self.df_raw[feature_cols].iloc[0].to_dict()

        processed_sample = pipeline.transform_single_sample(sample_dict)
        assert isinstance(processed_sample, np.ndarray)
        assert processed_sample.shape == (30,), f"Expected shape (30,), got {processed_sample.shape}"
        assert not np.isnan(processed_sample).any(), "Single sample transform contains NaNs."
        assert not np.isinf(processed_sample).any(), "Single sample transform contains Infs."

    def test_12_saved_preprocessing_reloaded_successfully(self):
        """12. Verify serialized preprocessing pipeline can be reloaded and transforms identically."""
        pipeline_orig = MedicalPreprocessingPipeline()
        X_train_raw, X_val_raw, X_test_raw, y_train, _, _ = split_data_leakage_safe(
            df=self.df_raw, target_column=TARGET_COLUMN, random_seed=RANDOM_SEED
        )
        pipeline_orig.fit(X_train_raw, y_train)

        # Reload from disk
        pipeline_loaded = MedicalPreprocessingPipeline.load(
            os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
        )

        out_orig = pipeline_orig.transform(X_test_raw).values
        out_loaded = pipeline_loaded.transform(X_test_raw).values

        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-7), (
            "Reloaded pipeline produced different output from original pipeline."
        )
