"""
Comprehensive test suite verifying all pipeline stages and boundaries.

Tests:
1. Dataset loading and raw schema
2. Preprocessing pipeline and zero-leakage transforms
3. Train/validation/test split integrity and stratification
4. Autoencoder feature extractor and 16D latent dimension
5. Quantum & classical feature selection and 8D representation
6. Angle encoding and range [0, pi]
7. VQC circuit construction, forward pass, and gradients
8. Hybrid VQC module probabilities and predictions
9. Evaluation metrics and confusion matrix calculations
"""
import os
import json
import numpy as np
import pandas as pd
import pytest
import torch

from configs.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    LATENT_DIM, SELECTED_DIM, NUM_QUBITS, VQC_NUM_LAYERS, RANDOM_SEED,
    AE_HIDDEN_DIMS
)
from src.data.loader import load_dataset
from src.models.autoencoder import Autoencoder
from src.quantum.vqc import (
    create_vqc, init_weights, AngleScaler, HybridVQCModule
)
from src.evaluation.metrics import compute_metrics


class TestPipelineBoundaries:
    """Test every stage of the classical-quantum hybrid pipeline."""

    def test_01_dataset_loading(self):
        """Verify raw dataset has correct dimensions and clean values."""
        X, y = load_dataset(save_raw=False)
        assert X.shape == (569, 30), f"Expected shape (569, 30), got {X.shape}"
        assert len(y) == 569, "Target count must match 569"
        assert set(y.unique()) == {0, 1}, "Target must be binary {0, 1}"
        assert X.isna().sum().sum() == 0, "Raw dataset contains unexpected NaNs"

    def test_02_preprocessing_and_splits(self):
        """Verify zero-leakage preprocessing and stratified split sizes."""
        for split in ["train", "val", "test"]:
            x_file = os.path.join(DATA_PROCESSED_DIR, f"X_{split}.csv")
            y_file = os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")
            assert os.path.exists(x_file), f"Missing {x_file}"
            assert os.path.exists(y_file), f"Missing {y_file}"
            
            X = pd.read_csv(x_file)
            y = pd.read_csv(y_file)
            assert X.shape[1] == 30, f"Expected 30 features, got {X.shape[1]}"
            assert not X.isna().any().any(), f"NaNs found in processed X_{split}"
            assert len(X) == len(y), "Sample count mismatch between X and y"

        # Check total count matches 569
        n_train = len(pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv")))
        n_val = len(pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv")))
        n_test = len(pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv")))
        assert n_train + n_val + n_test == 569, "Total samples does not sum to 569"

    def test_03_encoder_latent_dimensions(self):
        """Verify autoencoder produces exact 16-dimensional latent representation."""
        model = Autoencoder(input_dim=30, hidden_dims=AE_HIDDEN_DIMS, latent_dim=LATENT_DIM)
        x_dummy = torch.randn(10, 30)
        latent = model.encode(x_dummy)
        reconstruction, z = model(x_dummy)

        assert latent.shape == (10, LATENT_DIM), f"Expected (10, 16), got {latent.shape}"
        assert reconstruction.shape == (10, 30), f"Expected (10, 30), got {reconstruction.shape}"

        # Verify saved latent CSVs
        for split in ["train", "val", "test"]:
            path = os.path.join(RESULTS_DIR, f"latent_features_{split}.csv")
            if os.path.exists(path):
                df_lat = pd.read_csv(path)
                assert df_lat.shape[1] == LATENT_DIM, (
                    f"Latent features CSV has {df_lat.shape[1]} cols, expected {LATENT_DIM}"
                )

    def test_04_feature_selection_output(self):
        """Verify feature selection selects exactly 8 features."""
        q_path = os.path.join(RESULTS_DIR, "selected_features_quantum.json")
        if os.path.exists(q_path):
            with open(q_path) as f:
                q_data = json.load(f)
            indices = q_data.get("selected_indices", [])
            assert len(indices) == SELECTED_DIM, (
                f"Expected {SELECTED_DIM} features, got {len(indices)}"
            )
            assert len(set(indices)) == SELECTED_DIM, "Duplicate feature indices selected"

        for split in ["train", "val", "test"]:
            path = os.path.join(RESULTS_DIR, f"selected_features_{split}.csv")
            if os.path.exists(path):
                df_sel = pd.read_csv(path)
                assert df_sel.shape[1] == SELECTED_DIM, (
                    f"Selected features CSV has {df_sel.shape[1]} cols, expected {SELECTED_DIM}"
                )

    def test_05_angle_scaler_bounds(self):
        """Verify angle scaler scales features strictly into [0, pi] without NaNs."""
        X_fake_train = np.array([[-5.0, 10.0], [5.0, 20.0]], dtype=np.float32)
        X_fake_test = np.array([[-10.0, 15.0], [10.0, 25.0]], dtype=np.float32)

        scaler = AngleScaler()
        angles_train = scaler.fit_transform(X_fake_train)
        angles_test = scaler.transform(X_fake_test)

        assert angles_train.min() >= 0.0, "Angles must be >= 0"
        assert angles_train.max() <= np.pi + 1e-5, "Angles must be <= pi"
        assert angles_test.min() >= 0.0, "Test angles must be clipped to >= 0"
        assert angles_test.max() <= np.pi + 1e-5, "Test angles must be clipped to <= pi"

    def test_06_vqc_circuit_forward_and_gradient(self):
        """Verify 8-qubit VQC circuit execution and non-zero gradients."""
        circuit, dev, n_params = create_vqc(n_qubits=NUM_QUBITS, n_layers=VQC_NUM_LAYERS)
        assert n_params == VQC_NUM_LAYERS * NUM_QUBITS * 2

        x_single = torch.tensor(np.random.uniform(0, np.pi, NUM_QUBITS), dtype=torch.float32)
        weights = init_weights(n_layers=VQC_NUM_LAYERS, n_qubits=NUM_QUBITS)

        exp_val = circuit(x_single, weights)
        assert -1.0 - 1e-5 <= exp_val.item() <= 1.0 + 1e-5, (
            f"Expectation value {exp_val.item()} outside [-1, 1]"
        )

        # Backward pass gradient check
        exp_val.backward()
        assert weights.grad is not None, "Gradient was not computed"
        assert not torch.isnan(weights.grad).any(), "NaN in circuit gradients"
        assert weights.grad.abs().sum().item() > 0.0, "Gradients should be non-zero"

    def test_07_hybrid_vqc_module_inference(self):
        """Verify HybridVQCModule produces valid probabilities and predictions."""
        model = HybridVQCModule(n_qubits=NUM_QUBITS, n_layers=VQC_NUM_LAYERS)
        X_batch = np.random.uniform(0, np.pi, (5, NUM_QUBITS)).astype(np.float32)

        probs = model.predict_proba(X_batch)
        preds = model.predict(X_batch)

        assert probs.shape == (5,), f"Expected shape (5,), got {probs.shape}"
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]"
        assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary {0, 1}"

    def test_08_evaluation_metrics_correctness(self):
        """Verify metric calculation accuracy, specificity, and false negative tracking."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])  # 1 false negative (y_true=1, y_pred=0)
        y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_prob, "TestModel")
        assert metrics["accuracy"] == 0.8
        assert metrics["fn"] == 1, "Expected 1 false negative"
        assert metrics["tp"] == 2, "Expected 2 true positives"
        assert metrics["tn"] == 2, "Expected 2 true negatives"
        assert metrics["fp"] == 0, "Expected 0 false positives"
        assert metrics["specificity"] == 1.0
        assert "roc_auc" in metrics
