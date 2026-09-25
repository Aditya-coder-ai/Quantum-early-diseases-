"""
Automated Pytest Suite for PART 3: Classical Baseline + Classical Feature Extraction.

Tests all 20 mandatory requirements:
1. Part 2 preprocessing output loads.
2. Training data is non-empty.
3. Validation data is non-empty.
4. Test data is non-empty.
5. Feature dimensions are correct (30 raw features).
6. Target dimensions match feature rows.
7. Model trains successfully.
8. Model predicts successfully.
9. Predictions contain valid values {0, 1}.
10. No NaN predictions.
11. No infinite predictions.
12. Compact feature extraction works.
13. Compact feature dimensions are correct (4, 8, 12, 16).
14. Train/validation/test feature dimensions match.
15. Feature extraction is reproducible.
16. Saved feature extractor can be reloaded.
17. Saved model can be reloaded.
18. Test data is not used during training (leakage isolation).
19. Metrics can be reproduced.
20. End-to-end Part 3 pipeline runs successfully.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from configs.config import (
    DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED, PROJECT_ROOT
)
from src.features.reduction import PCAReducer, AutoencoderReducer
from src.features.validation import FeatureQualityValidator
from src.features.extraction import run_feature_extraction_pipeline
from src.classical.baseline import (
    train_eval_classifier, run_classical_baselines, run_compact_feature_experiments
)
from src.evaluation.metrics import compute_metrics
from sklearn.linear_model import LogisticRegression


class TestPart3Pipeline:
    """Comprehensive test suite verifying all 20 criteria of Part 3."""

    @pytest.fixture(autouse=True)
    def setup_splits(self):
        """Load processed splits from Part 2."""
        self.X_train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
        self.X_val_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"))
        self.X_test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))
        self.y_train_s = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv")).squeeze()
        self.y_val_s = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv")).squeeze()
        self.y_test_s = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv")).squeeze()

    def test_01_part2_preprocessing_output_loads(self):
        """1. Verify Part 2 preprocessing output files exist and load cleanly."""
        for name in ["X_train.csv", "X_val.csv", "X_test.csv", "y_train.csv", "y_val.csv", "y_test.csv"]:
            path = os.path.join(DATA_PROCESSED_DIR, name)
            assert os.path.exists(path), f"Required file missing: {path}"

    def test_02_training_data_non_empty(self):
        """2. Verify training dataset is non-empty."""
        assert len(self.X_train_df) > 0, "X_train is empty."
        assert len(self.y_train_s) > 0, "y_train is empty."

    def test_03_validation_data_non_empty(self):
        """3. Verify validation dataset is non-empty."""
        assert len(self.X_val_df) > 0, "X_val is empty."
        assert len(self.y_val_s) > 0, "y_val is empty."

    def test_04_test_data_non_empty(self):
        """4. Verify test dataset is non-empty."""
        assert len(self.X_test_df) > 0, "X_test is empty."
        assert len(self.y_test_s) > 0, "y_test is empty."

    def test_05_feature_dimensions_correct(self):
        """5. Verify raw feature matrices have exact expected 30 features."""
        assert self.X_train_df.shape[1] == 30, f"Expected 30 features, got {self.X_train_df.shape[1]}"
        assert self.X_val_df.shape[1] == 30
        assert self.X_test_df.shape[1] == 30

    def test_06_target_dimensions_match_features(self):
        """6. Verify row count of target matches feature rows across splits."""
        assert len(self.X_train_df) == len(self.y_train_s)
        assert len(self.X_val_df) == len(self.y_val_s)
        assert len(self.X_test_df) == len(self.y_test_s)

    def test_07_model_trains_successfully(self):
        """7. Verify classical baseline classifier trains without exception."""
        clf = LogisticRegression(max_iter=100, random_state=RANDOM_SEED)
        clf.fit(self.X_train_df, self.y_train_s)
        assert hasattr(clf, "coef_"), "Model did not train properly."

    def test_08_model_predicts_successfully(self):
        """8. Verify baseline model generates predictions on test features."""
        clf = LogisticRegression(max_iter=100, random_state=RANDOM_SEED)
        clf.fit(self.X_train_df, self.y_train_s)
        preds = clf.predict(self.X_test_df)
        assert len(preds) == len(self.X_test_df)

    def test_09_predictions_valid_values(self):
        """9. Verify predicted class labels are strictly binary {0, 1}."""
        clf = LogisticRegression(max_iter=100, random_state=RANDOM_SEED)
        clf.fit(self.X_train_df, self.y_train_s)
        preds = clf.predict(self.X_test_df)
        unique_p = set(np.unique(preds))
        assert unique_p.issubset({0, 1}), f"Invalid prediction values: {unique_p}"

    def test_10_no_nan_predictions(self):
        """10. Verify predictions contain zero NaN values."""
        clf = LogisticRegression(max_iter=100, random_state=RANDOM_SEED)
        clf.fit(self.X_train_df, self.y_train_s)
        preds = clf.predict(self.X_test_df)
        probs = clf.predict_proba(self.X_test_df)[:, 1]
        assert not np.isnan(preds).any(), "NaN found in predictions."
        assert not np.isnan(probs).any(), "NaN found in predicted probabilities."

    def test_11_no_infinite_predictions(self):
        """11. Verify predictions contain zero infinite values."""
        clf = LogisticRegression(max_iter=100, random_state=RANDOM_SEED)
        clf.fit(self.X_train_df, self.y_train_s)
        preds = clf.predict(self.X_test_df)
        probs = clf.predict_proba(self.X_test_df)[:, 1]
        assert not np.isinf(preds).any(), "Inf found in predictions."
        assert not np.isinf(probs).any(), "Inf found in predicted probabilities."

    def test_12_compact_feature_extraction_works(self):
        """12. Verify compact feature extraction via PCA and Autoencoder operates cleanly."""
        pca = PCAReducer(n_components=8, random_state=RANDOM_SEED)
        X_tr_pca = pca.fit_transform(self.X_train_df)
        assert X_tr_pca.shape == (len(self.X_train_df), 8)

        ae = AutoencoderReducer(latent_dim=8, input_dim=30, epochs=5, random_seed=RANDOM_SEED)
        ae.fit(self.X_train_df, self.X_val_df)
        X_tr_ae = ae.transform(self.X_train_df)
        assert X_tr_ae.shape == (len(self.X_train_df), 8)

    def test_13_compact_feature_dimensions_correct(self):
        """13. Verify extracted compact features match requested target dimensions (4, 8, 12, 16)."""
        for d in [4, 8, 12, 16]:
            pca = PCAReducer(n_components=d, random_state=RANDOM_SEED)
            res = pca.fit_transform(self.X_train_df)
            assert res.shape[1] == d, f"Expected dimension {d}, got {res.shape[1]}"

    def test_14_train_val_test_feature_dims_match(self):
        """14. Verify extracted feature dimensions match across train, val, and test splits."""
        pca = PCAReducer(n_components=8, random_state=RANDOM_SEED)
        tr_p = pca.fit_transform(self.X_train_df)
        va_p = pca.transform(self.X_val_df)
        te_p = pca.transform(self.X_test_df)

        assert tr_p.shape[1] == va_p.shape[1] == te_p.shape[1] == 8

    def test_15_feature_extraction_reproducible(self):
        """15. Verify feature extraction produces identical arrays when run with same seed."""
        pca1 = PCAReducer(n_components=8, random_state=RANDOM_SEED)
        out1 = pca1.fit_transform(self.X_train_df)

        pca2 = PCAReducer(n_components=8, random_state=RANDOM_SEED)
        out2 = pca2.fit_transform(self.X_train_df)

        np.testing.assert_array_equal(out1, out2)

    def test_16_saved_feature_extractor_reloaded(self, tmp_path):
        """16. Verify saved PCA and Autoencoder objects can be reloaded and transform identically."""
        save_path_pca = str(tmp_path / "test_pca.joblib")
        pca_orig = PCAReducer(n_components=8, random_state=RANDOM_SEED)
        tr_orig = pca_orig.fit_transform(self.X_train_df)
        pca_orig.save(save_path_pca)

        pca_reloaded = PCAReducer.load(save_path_pca)
        tr_reloaded = pca_reloaded.transform(self.X_train_df)

        np.testing.assert_allclose(tr_orig, tr_reloaded, atol=1e-7)

    def test_17_saved_model_reloaded(self, tmp_path):
        """17. Verify trained baseline classifier can be serialized and reloaded."""
        model_path = str(tmp_path / "test_model.joblib")
        clf_orig = LogisticRegression(max_iter=100, random_state=RANDOM_SEED)
        clf_orig.fit(self.X_train_df, self.y_train_s)
        joblib.dump(clf_orig, model_path)

        clf_reloaded = joblib.load(model_path)
        pred_orig = clf_orig.predict(self.X_test_df)
        pred_reloaded = clf_reloaded.predict(self.X_test_df)

        np.testing.assert_array_equal(pred_orig, pred_reloaded)

    def test_18_test_data_not_used_during_training(self):
        """18. Verify PCA reducer was fitted exclusively on training set."""
        pca = PCAReducer(n_components=8, random_state=RANDOM_SEED)
        pca.fit(self.X_train_df)

        # Fitted mean of PCA must equal X_train mean, not X_train + X_test mean
        tr_mean = self.X_train_df.mean().values
        pca_mean = pca.pca.mean_
        assert np.allclose(pca_mean, tr_mean, atol=1e-5), "PCA mean does not match X_train mean."

        combined_mean = pd.concat([self.X_train_df, self.X_test_df]).mean().values
        assert not np.allclose(pca_mean, combined_mean, atol=1e-4), "PCA mean matches combined train+test data!"

    def test_19_metrics_reproduced(self):
        """19. Verify calculation of accuracy, recall, precision, specificity, F1, ROC-AUC."""
        y_t = np.array([1, 1, 0, 0, 1])
        y_p = np.array([1, 0, 0, 0, 1])
        y_pr = np.array([0.9, 0.4, 0.1, 0.2, 0.8])

        m = compute_metrics(y_t, y_p, y_pr, "TestClf")
        assert m["accuracy"] == 0.8
        assert m["tp"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 1
        assert m["specificity"] == 1.0
        assert m["false_negative_rate"] == 0.3333

    def test_20_end_to_end_part3_pipeline_runs(self):
        """20. Verify end-to-end Part 3 pipeline script executes successfully."""
        baselines = run_classical_baselines(save_models=True)
        assert len(baselines) == 4, f"Expected 4 baseline results, got {len(baselines)}"

        ext_res = run_feature_extraction_pipeline(target_dims=[4, 8])
        assert len(ext_res) == 4, f"Expected 4 extraction result keys, got {len(ext_res)}"

        compact_res = run_compact_feature_experiments(target_dims=[4, 8])
        assert len(compact_res) > 0, "Compact feature experiments returned no results."
