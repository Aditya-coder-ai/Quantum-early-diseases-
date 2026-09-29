"""
Automated Test Suite for Part 6 — Variational Quantum Classifier (VQC).
Tests 20 critical points covering:
    1. Feature dimension matching
    2. Feature-to-qubit mapping
    3. Scaler training-only fitting
    4. Encoded values finiteness & bounds
    5. Circuit construction
    6. Circuit wire counts
    7. Circuit output dimensions
    8. Trainable parameter counts
    9. Gradient existence
    10. Gradient finiteness
    11. Parameter updating
    12. Loss finiteness
    13. Valid prediction classes
    14. Probability range constraints
    15. Checkpoint serialization
    16. Checkpoint deserialization & state restoration
    17. Deterministic reproducibility
    18. Evaluation shape preservation
    19. Test set leakage protection
    20. End-to-end tiny VQC execution
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import DATA_PROCESSED_DIR
from src.vqc.config import QAOA_FEATURES_DIR, VQCConfig
from src.vqc.encoding import AngleScaler, FeatureQubitMapper
from src.vqc.circuit import create_vqc_circuit, count_quantum_resources
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.training import train_vqc, compute_weighted_bce_loss, run_tiny_vqc_test
from src.vqc.evaluation import compute_comprehensive_metrics, evaluate_vqc_model


# ── Test 1: Feature dimension matches qubit count ────────────────────────────
def test_01_feature_dim_matches_qubit_count():
    mapper = FeatureQubitMapper([f"f_{i}" for i in range(8)])
    assert mapper.n_features == 8
    X = np.random.randn(10, 8)
    X_sliced, names = mapper.slice_features(X, 4)
    assert X_sliced.shape[1] == 4
    assert len(names) == 4


# ── Test 2: Feature-to-qubit mapping is deterministic ────────────────────────
def test_02_feature_qubit_mapping_correct():
    names = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area"]
    mapper = FeatureQubitMapper(names)
    for i, name in enumerate(names):
        assert mapper.get_qubit_for_feature(i) == i
        assert mapper.get_feature_name(i) == name


# ── Test 3: Scaler is training-only (zero data leakage) ───────────────────────
def test_03_scaler_is_training_only():
    scaler = AngleScaler()
    assert not scaler.is_fitted
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(np.zeros((5, 4)))

    X_train = np.array([[0.0, 10.0], [5.0, 20.0]])
    scaler.fit(X_train)
    assert scaler.is_fitted
    assert np.allclose(scaler.min_vals, [0.0, 10.0])
    assert np.allclose(scaler.max_vals, [5.0, 20.0])


# ── Test 4: Encoded values are finite and strictly in [0, pi] ────────────────
def test_04_encoded_values_bounds():
    scaler = AngleScaler(target_range=(0.0, np.pi))
    X_train = np.random.randn(20, 8)
    angles_tr = scaler.fit_transform(X_train)

    assert not np.isnan(angles_tr).any()
    assert not np.isinf(angles_tr).any()
    assert (angles_tr >= 0.0).all()
    assert (angles_tr <= np.pi + 1e-5).all()

    # Out-of-bounds validation sample should be clipped to [0, pi]
    X_ext = np.array([[-1000.0] * 8, [1000.0] * 8])
    angles_ext = scaler.transform(X_ext)
    assert (angles_ext >= 0.0).all()
    assert (angles_ext <= np.pi + 1e-5).all()


# ── Test 5: Circuit construction succeeds ────────────────────────────────────
def test_05_circuit_construction_succeeds():
    qnode, dev, n_params, res = create_vqc_circuit(n_qubits=4, n_layers=2)
    assert qnode is not None
    assert dev is not None
    assert n_params > 0


# ── Test 6: Circuit qubit count is correct ───────────────────────────────────
def test_06_circuit_qubit_count_correct():
    for n in [4, 6, 8]:
        res = count_quantum_resources(n_qubits=n, n_layers=2)
        assert res["n_qubits"] == n


# ── Test 7: Circuit output dimension is valid scalar / batch ─────────────────
def test_07_circuit_output_dimension():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    x_single = torch.tensor(np.random.uniform(0, np.pi, 4), dtype=torch.float32)
    out_single = model(x_single)
    assert out_single.shape == (1,)

    x_batch = torch.tensor(np.random.uniform(0, np.pi, (5, 4)), dtype=torch.float32)
    out_batch = model(x_batch)
    assert out_batch.shape == (5,)


# ── Test 8: Trainable parameter count is mathematically correct ──────────────
def test_08_trainable_parameter_count():
    # Ansatz A (RY only): n_layers * n_qubits * 1
    res_a = count_quantum_resources(n_qubits=8, n_layers=2, ansatz_type="ansatz_a")
    assert res_a["trainable_parameters"] == 2 * 8 * 1

    # Ansatz B (RY + RZ): n_layers * n_qubits * 2
    res_b = count_quantum_resources(n_qubits=8, n_layers=2, ansatz_type="ansatz_b")
    assert res_b["trainable_parameters"] == 2 * 8 * 2


# ── Test 9: Gradients exist on trainable parameters ──────────────────────────
def test_09_gradients_exist():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    x = torch.tensor(np.random.uniform(0, np.pi, (4, 4)), dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

    logits = model(x)
    loss = compute_weighted_bce_loss(logits, y)
    loss.backward()

    assert model.weights.grad is not None
    assert model.head.weight.grad is not None


# ── Test 10: Gradients are strictly finite (no NaN, no Inf) ──────────────────
def test_10_gradients_are_finite():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    x = torch.tensor(np.random.uniform(0, np.pi, (4, 4)), dtype=torch.float32)
    y = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    logits = model(x)
    loss = compute_weighted_bce_loss(logits, y)
    loss.backward()

    assert not torch.isnan(model.weights.grad).any()
    assert not torch.isinf(model.weights.grad).any()


# ── Test 11: Parameters update after optimizer step ──────────────────────────
def test_11_parameters_update():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    w_initial = model.weights.clone().detach()

    x = torch.tensor(np.random.uniform(0, np.pi, (4, 4)), dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss = compute_weighted_bce_loss(model(x), y)
    loss.backward()
    optimizer.step()

    diff = (model.weights - w_initial).abs().sum().item()
    assert diff > 1e-4


# ── Test 12: Loss is finite ──────────────────────────────────────────────────
def test_12_loss_is_finite():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    x = torch.tensor(np.random.uniform(0, np.pi, (4, 4)), dtype=torch.float32)
    y = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    loss = compute_weighted_bce_loss(model(x), y)
    assert torch.isfinite(loss)


# ── Test 13: Prediction values are binary in {0, 1} ──────────────────────────
def test_13_prediction_values_are_binary():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    x = np.random.uniform(0, np.pi, (10, 4)).astype(np.float32)
    preds = model.predict(x)
    assert set(np.unique(preds)).issubset({0, 1})


# ── Test 14: Probability conversion is strictly in [0, 1] ────────────────────
def test_14_probability_range():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    x = np.random.uniform(0, np.pi, (10, 4)).astype(np.float32)
    probs = model.predict_proba(x)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


# ── Test 15 & 16: Model checkpoint serialization & restoration ───────────────
def test_15_16_checkpoint_save_and_reload():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    x = np.random.uniform(0, np.pi, (5, 4)).astype(np.float32)
    initial_probs = model.predict_proba(x)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "vqc_temp.pt")
        model.save(path)
        assert os.path.exists(path)

        reloaded = VariationalQuantumClassifier.load(path)
        reloaded_probs = reloaded.predict_proba(x)
        assert np.allclose(initial_probs, reloaded_probs, atol=1e-5)


# ── Test 17: Deterministic reproducibility with same seed ────────────────────
def test_17_deterministic_reproducibility():
    m1 = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=123)
    m2 = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=123)
    assert torch.allclose(m1.weights, m2.weights)


# ── Test 18: Validation and test shapes remain preserved ──────────────────────
def test_18_validation_test_shapes_preserved():
    X_val = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_val_selected.csv"))
    X_test = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_test_selected.csv"))
    y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv"))

    assert X_val.shape == (85, 8)
    assert X_test.shape == (86, 8)
    assert len(y_val) == 85
    assert len(y_test) == 86


# ── Test 19: Test set leakage protection (zero test data in scaler/model) ────
def test_19_test_leakage_audit():
    X_train = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_train_selected.csv")).values
    X_test = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_test_selected.csv")).values

    scaler = AngleScaler().fit(X_train)
    # Confirm scaler parameters match X_train exactly, not X_test
    assert np.allclose(scaler.min_vals, X_train.min(axis=0))
    assert np.allclose(scaler.max_vals, X_train.max(axis=0))
    assert not np.allclose(scaler.min_vals, X_test.min(axis=0))


# ── Test 20: End-to-end tiny VQC execution ───────────────────────────────────
def test_20_end_to_end_tiny_vqc_test():
    res = run_tiny_vqc_test(seed=42)
    assert res["status"] == "PASSED"
    assert res["has_finite_grad"] is True
    assert res["params_updated"] is True
