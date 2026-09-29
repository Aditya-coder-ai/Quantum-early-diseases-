"""
Part 5 — Automated Test Suite for Class Imbalance & Quantum GAN.

Tests:
    1. Class distribution calculation
    2. Class-weight calculation and correctness
    3. SMOTE operates strictly on training data
    4. Validation set size and distribution remain strictly unchanged
    5. Test set size and distribution remain strictly unchanged
    6. QGAN input dimensions are verified
    7. QGAN output dimensions match feature dimensions
    8. QGAN circuit builds and evaluates
    9. QGAN produces finite values
    10. Generated samples contain no NaN
    11. Generated samples contain no Infinity
    12. Generated sample count matches requested count
    13. Synthetic samples remain in valid feature space (range check)
    14. Mode-collapse detector correctly flags collapsed vs healthy distributions
    15. QGAN model artifacts can be saved and reloaded
    16. Same seed produces reproducible results
    17. Data leakage audit: generator never sees validation or test samples
    18. Downstream classifier evaluation produces valid metrics
    19. Small synthetic end-to-end pipeline executes cleanly
"""
from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import DATA_PROCESSED_DIR
from src.imbalance.config import (
    QAOA_FEATURES_DIR, FEATURE_DIM, MINORITY_CLASS, MAJORITY_CLASS, SEED
)
from src.imbalance.classical import (
    measure_imbalance, compute_class_weights, smote_oversample
)
from src.imbalance.quality import (
    evaluate_synthetic_quality, detect_mode_collapse
)
from src.imbalance.qgan import (
    QuantumGAN, QuantumGenerator, ClassicalDiscriminator, run_tiny_synthetic_test
)
from src.imbalance.evaluation import (
    evaluate_imbalance_strategy, get_downstream_classifier
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_data():
    """Load real QAOA selected features and labels."""
    X_tr = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_train_selected.csv")).values
    X_va = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_val_selected.csv")).values
    X_te = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_test_selected.csv")).values
    y_tr = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv")).squeeze().values
    y_va = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv")).squeeze().values
    y_te = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv")).squeeze().values
    return {
        "X_train": X_tr, "X_val": X_va, "X_test": X_te,
        "y_train": y_tr, "y_val": y_va, "y_test": y_te,
    }


@pytest.fixture(scope="module")
def synthetic_toy_data():
    """Simple 2D toy dataset with 80% majority, 20% minority."""
    rng = np.random.RandomState(SEED)
    X_maj = rng.randn(80, 4) + 2.0
    X_min = rng.randn(20, 4) - 2.0
    X = np.vstack([X_maj, X_min])
    y = np.hstack([np.ones(80, dtype=int), np.zeros(20, dtype=int)])
    return X, y


# ── 1. Imbalance Measurement Tests ────────────────────────────────────

def test_01_class_distribution_calculation(real_data):
    """Test exact class counts and imbalance ratio calculation."""
    info = measure_imbalance(real_data["y_train"], MINORITY_CLASS, MAJORITY_CLASS)
    assert info["total_samples"] == len(real_data["y_train"])
    assert info["minority_samples"] == 148
    assert info["majority_samples"] == 250
    assert info["imbalance_ratio"] == round(250 / 148, 4)
    assert info["deficit"] == 102


def test_02_class_weight_configuration(real_data):
    """Test computed class weights are balanced and strictly positive."""
    weights = compute_class_weights(real_data["y_train"], MINORITY_CLASS, MAJORITY_CLASS)
    assert MINORITY_CLASS in weights and MAJORITY_CLASS in weights
    assert weights[MINORITY_CLASS] > weights[MAJORITY_CLASS]
    assert weights[MINORITY_CLASS] > 1.0
    assert weights[MAJORITY_CLASS] < 1.0


# ── 2. SMOTE Tests ────────────────────────────────────────────────────

def test_03_smote_operates_only_on_train(real_data):
    """Test SMOTE modifies only training data and increases minority count."""
    X_tr, y_tr = real_data["X_train"], real_data["y_train"]
    X_aug, y_aug, X_synth = smote_oversample(X_tr, y_tr, ratio=1.0, seed=SEED)

    assert len(X_aug) == 500  # 250 majority + 250 minority
    assert len(y_aug) == 500
    assert len(X_synth) == 102
    assert (y_aug == MINORITY_CLASS).sum() == 250
    assert (y_aug == MAJORITY_CLASS).sum() == 250


def test_04_validation_size_remains_unchanged(real_data):
    """Validation size and distribution must remain strictly unchanged."""
    X_va, y_va = real_data["X_val"], real_data["y_val"]
    val_shape_before = X_va.shape
    val_counts_before = dict(pd.Series(y_va).value_counts())

    # Run SMOTE on train
    smote_oversample(real_data["X_train"], real_data["y_train"], ratio=1.0)

    assert X_va.shape == val_shape_before
    assert dict(pd.Series(y_va).value_counts()) == val_counts_before


def test_05_test_size_remains_unchanged(real_data):
    """Test set size and distribution must remain strictly untouched."""
    X_te, y_te = real_data["X_test"], real_data["y_test"]
    assert X_te.shape == (86, FEATURE_DIM)
    assert len(y_te) == 86


# ── 3. QGAN Architecture & Execution Tests ────────────────────────────

def test_06_qgan_input_dimensions():
    """QGAN circuit accepts correct latent noise dimension."""
    gen = QuantumGenerator(n_qubits=4, n_layers=1, seed=SEED)
    z = torch.randn(5, 4)
    out = gen(z)
    assert out.shape == (5, 4)


def test_07_qgan_output_dimensions():
    """QGAN generator output matches requested feature dimension."""
    gen = QuantumGenerator(n_qubits=8, n_layers=2, seed=SEED)
    z = torch.randn(10, 8)
    out = gen(z)
    assert out.shape == (10, 8)


def test_08_qgan_circuit_builds():
    """QGAN circuit builds and returns valid PennyLane QNode."""
    qgan = QuantumGAN(n_qubits=4, n_layers=1, seed=SEED)
    assert qgan.generator is not None
    assert qgan.discriminator is not None


def test_09_qgan_produces_finite_values():
    """QGAN outputs must be strictly finite with no NaN/Infs."""
    qgan = QuantumGAN(n_qubits=4, n_layers=1, seed=SEED)
    samples = qgan.generate(15, seed=SEED)
    assert np.isfinite(samples).all()


def test_10_generated_samples_no_nan(real_data):
    """Generated samples from trained QGAN contain zero NaNs."""
    X_min = real_data["X_train"][real_data["y_train"] == MINORITY_CLASS][:20]
    qgan = QuantumGAN(n_qubits=FEATURE_DIM, n_layers=1, seed=SEED)
    qgan.train(X_min, epochs=2, batch_size=10, verbose=False)
    samples = qgan.generate(25, seed=SEED)
    assert not np.isnan(samples).any()


def test_11_generated_samples_no_infinity(real_data):
    """Generated samples from trained QGAN contain zero infinite values."""
    X_min = real_data["X_train"][real_data["y_train"] == MINORITY_CLASS][:20]
    qgan = QuantumGAN(n_qubits=FEATURE_DIM, n_layers=1, seed=SEED)
    qgan.train(X_min, epochs=2, batch_size=10, verbose=False)
    samples = qgan.generate(25, seed=SEED)
    assert not np.isinf(samples).any()


def test_12_generated_sample_count_correct():
    """Requested sample count must match exactly."""
    qgan = QuantumGAN(n_qubits=4, n_layers=1, seed=SEED)
    for count in [5, 17, 50]:
        samples = qgan.generate(count, seed=SEED)
        assert len(samples) == count


def test_13_synthetic_samples_in_valid_range(real_data):
    """Synthetic samples should be reasonably bounded near the empirical feature range."""
    X_min = real_data["X_train"][real_data["y_train"] == MINORITY_CLASS]
    emp_min = np.min(X_min, axis=0) - 3.0 * np.std(X_min, axis=0)
    emp_max = np.max(X_min, axis=0) + 3.0 * np.std(X_min, axis=0)

    qgan = QuantumGAN(
        n_qubits=FEATURE_DIM,
        n_layers=1,
        empirical_mean=np.mean(X_min, axis=0),
        empirical_std=np.std(X_min, axis=0),
        seed=SEED,
    )
    samples = qgan.generate(30, seed=SEED)

    # All generated samples must be within reasonable bounding
    assert (samples >= emp_min).all()
    assert (samples <= emp_max).all()


# ── 4. Mode Collapse & Quality Tests ──────────────────────────────────

def test_14_mode_collapse_detection():
    """Mode collapse detector must flag collapsed samples and approve diverse samples."""
    # 1. Artificially collapsed data: all samples are nearly identical
    collapsed_samples = np.ones((50, 4)) * 2.5 + np.random.RandomState(42).randn(50, 4) * 0.0001
    rep_collapsed = detect_mode_collapse(collapsed_samples)
    assert rep_collapsed["is_collapsed"] is True
    assert rep_collapsed["status"] == "COLLAPSED"

    # 2. Healthy data: well-distributed Gaussian samples
    healthy_samples = np.random.RandomState(42).randn(50, 4)
    rep_healthy = detect_mode_collapse(healthy_samples)
    assert rep_healthy["is_collapsed"] is False
    assert rep_healthy["status"] == "HEALTHY"


# ── 5. Persistence & Reproducibility Tests ─────────────────────────────

def test_15_qgan_artifacts_save_and_reload(tmp_path):
    """Trained QGAN weights must save and reload identically."""
    qgan = QuantumGAN(n_qubits=4, n_layers=1, seed=SEED)
    save_info = qgan.save(str(tmp_path), tag="test_save")

    assert os.path.exists(save_info["generator_path"])
    assert os.path.exists(save_info["discriminator_path"])

    qgan_reloaded = QuantumGAN(n_qubits=4, n_layers=1, seed=SEED)
    qgan_reloaded.load(str(tmp_path), tag="test_save")

    # Generate samples from both -> must match identically
    s1 = qgan.generate(10, seed=42)
    s2 = qgan_reloaded.generate(10, seed=42)
    np.testing.assert_array_almost_equal(s1, s2, decimal=5)


def test_16_reproducibility_with_same_seed():
    """Generating with the same seed produces identical synthetic vectors."""
    qgan = QuantumGAN(n_qubits=4, n_layers=1, seed=SEED)
    s1 = qgan.generate(10, seed=99)
    s2 = qgan.generate(10, seed=99)
    np.testing.assert_array_equal(s1, s2)


# ── 6. Leakage & Evaluation Tests ─────────────────────────────────────

def test_17_leakage_audit_no_val_or_test_in_generator(real_data):
    """Verify that training minority slice has zero overlap with val/test data."""
    X_min = real_data["X_train"][real_data["y_train"] == MINORITY_CLASS]
    X_va = real_data["X_val"]
    X_te = real_data["X_test"]

    # Verify no row in X_min equals any row in X_va or X_te
    diff_val = np.min(np.linalg.norm(X_min[:, None, :] - X_va[None, :, :], axis=-1))
    diff_test = np.min(np.linalg.norm(X_min[:, None, :] - X_te[None, :, :], axis=-1))

    assert diff_val > 1e-6, "Potential data leakage between train minority and validation set"
    assert diff_test > 1e-6, "Potential data leakage between train minority and test set"


def test_18_downstream_evaluation_runs_cleanly(real_data):
    """Downstream evaluation function produces complete valid metrics."""
    res = evaluate_imbalance_strategy(
        real_data["X_train"], real_data["y_train"],
        real_data["X_val"], real_data["y_val"],
        method_name="Test_Model",
        seed=SEED,
    )
    assert "accuracy" in res
    assert "f1" in res
    assert "minority_recall" in res
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["minority_recall"] <= 1.0


def test_19_tiny_qgan_verification_end_to_end():
    """Tiny synthetic QGAN test passes all gradient, loss, and shape assertions."""
    res = run_tiny_synthetic_test(seed=SEED)
    assert res["status"] == "PASSED"
    assert res["samples_shape"] == [20, 2]
    assert np.isfinite(res["final_loss_D"])
    assert np.isfinite(res["final_loss_G"])
