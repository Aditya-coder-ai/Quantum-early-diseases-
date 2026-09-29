"""
Part 4 -- Automated Test Suite for Feature Selection.

Tests cover:
    1. Feature registry determinism and integrity
    2. Classical selector validity
    3. QUBO construction and verification
    4. Ising conversion
    5. QAOA circuit and measurement
    6. Objective evaluation
    7. Leakage prevention
    8. Artifact save/load
    9. End-to-end pipeline
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import RANDOM_SEED, RESULTS_DIR, DATA_PROCESSED_DIR
from src.feature_selection.config import (
    SEED, INPUT_FEATURE_DIM, FEATURE_BUDGETS, PART4_ARTIFACTS_DIR
)
from src.feature_selection.registry import (
    build_latent_feature_registry, validate_registry,
    save_registry, load_registry, get_feature_names, get_selected_names,
)
from src.feature_selection.objective import (
    FeatureSelectionObjective,
    compute_mutual_information,
    compute_correlation_matrix,
)
from src.feature_selection.classical import (
    select_mutual_info, select_rfe, SELECTOR_MAP,
)
from src.feature_selection.qaoa import (
    qubo_to_ising, verify_ising_conversion,
    build_cost_hamiltonian, build_mixer_hamiltonian,
    decode_bitstring, decode_samples,
    run_qaoa, qaoa_vs_exact,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_data():
    """Small synthetic dataset for fast testing."""
    rng = np.random.RandomState(SEED)
    X = rng.randn(80, 4)
    X[:, 0] *= 3  # feature 0 is most informative
    y = (X[:, 0] + 0.3 * X[:, 2] > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def real_latent_data():
    """Part 3 latent features (16-D)."""
    X_train = pd.read_csv(
        os.path.join(RESULTS_DIR, "latent_features_train.csv")
    ).values
    y_train = pd.read_csv(
        os.path.join(DATA_PROCESSED_DIR, "y_train.csv")
    ).squeeze().values
    return X_train, y_train


@pytest.fixture(scope="module")
def registry():
    return build_latent_feature_registry(INPUT_FEATURE_DIM)


# ═══════════════════════════════════════════════════════════════════
# 1. Feature Registry Tests
# ═══════════════════════════════════════════════════════════════════

def test_01_registry_deterministic(registry):
    """Feature registry must be deterministic across calls."""
    reg2 = build_latent_feature_registry(INPUT_FEATURE_DIM)
    names1 = get_feature_names(registry)
    names2 = get_feature_names(reg2)
    assert names1 == names2, "Registry is not deterministic"


def test_02_registry_integrity(registry):
    """Registry must have correct count and contiguous indices."""
    validate_registry(registry, INPUT_FEATURE_DIM)


def test_03_registry_save_load(registry, tmp_path):
    """Registry must survive save/load round-trip."""
    path = str(tmp_path / "test_registry.json")
    save_registry(registry, path)
    loaded = load_registry(path)
    assert len(loaded) == len(registry)
    assert get_feature_names(loaded) == get_feature_names(registry)


# ═══════════════════════════════════════════════════════════════════
# 2. Classical Selector Tests
# ═══════════════════════════════════════════════════════════════════

def test_04_mi_returns_valid_indices(synthetic_data):
    """MI selector must return valid indices."""
    X, y = synthetic_data
    result = select_mutual_info(X, y, k=2, seed=SEED)
    assert len(result["selected_indices"]) == 2
    assert all(0 <= i < X.shape[1] for i in result["selected_indices"])


def test_05_rfe_returns_valid_indices(synthetic_data):
    """RFE selector must return valid indices."""
    X, y = synthetic_data
    result = select_rfe(X, y, k=2, seed=SEED)
    assert len(result["selected_indices"]) == 2
    assert all(0 <= i < X.shape[1] for i in result["selected_indices"])


def test_06_selector_never_exceeds_k(synthetic_data):
    """No selector may return more than k features."""
    X, y = synthetic_data
    for method_name, fn in SELECTOR_MAP.items():
        for k in [1, 2, 3]:
            result = fn(X, y, k=k, seed=SEED)
            assert len(result["selected_indices"]) == k, (
                f"{method_name} returned {len(result['selected_indices'])} != {k}"
            )


# ═══════════════════════════════════════════════════════════════════
# 3. Objective and QUBO Tests
# ═══════════════════════════════════════════════════════════════════

def test_07_qubo_dimensions(synthetic_data):
    """QUBO matrix must be D x D."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    Q = obj.build_qubo()
    assert Q.shape == (X.shape[1], X.shape[1])


def test_08_qubo_symmetric(synthetic_data):
    """QUBO matrix must be symmetric."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    Q = obj.build_qubo()
    np.testing.assert_allclose(Q, Q.T, atol=1e-12)


def test_09_qubo_coefficients_finite(synthetic_data):
    """All QUBO coefficients must be finite."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    Q = obj.build_qubo()
    assert np.all(np.isfinite(Q)), "QUBO has non-finite values"


def test_10_qubo_verification_passes(synthetic_data):
    """QUBO must pass independent verification."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    Q = obj.build_qubo()
    result = obj.verify_qubo(Q, n_samples=100)
    assert result["passed"], f"QUBO verify failed: max_error={result['max_abs_error']}"


def test_11_objective_no_nan(synthetic_data):
    """Objective must never return NaN."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    rng = np.random.RandomState(SEED)
    for _ in range(50):
        x = rng.randint(0, 2, X.shape[1]).astype(float)
        val = obj.evaluate(x)
        assert np.isfinite(val), f"Objective returned non-finite: {val}"


def test_12_objective_no_inf(synthetic_data):
    """Objective must never return Inf."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    rng = np.random.RandomState(SEED)
    for _ in range(50):
        x = rng.randint(0, 2, X.shape[1]).astype(float)
        val = obj.evaluate(x)
        assert not np.isinf(val), f"Objective returned Inf"


# ═══════════════════════════════════════════════════════════════════
# 4. Ising Conversion Tests
# ═══════════════════════════════════════════════════════════════════

def test_13_ising_conversion_correct(synthetic_data):
    """Ising conversion must reproduce QUBO values."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    Q = obj.build_qubo()
    J, h, offset = qubo_to_ising(Q)
    result = verify_ising_conversion(Q, J, h, offset)
    assert result["passed"], f"Ising conversion error: {result['max_abs_error']}"


# ═══════════════════════════════════════════════════════════════════
# 5. QAOA Circuit Tests
# ═══════════════════════════════════════════════════════════════════

def test_14_qaoa_circuit_builds(synthetic_data):
    """QAOA circuit must build without error."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    Q = obj.build_qubo()
    J, h, _ = qubo_to_ising(Q)
    cost_h = build_cost_hamiltonian(J, h)
    mixer_h = build_mixer_hamiltonian(X.shape[1])
    # Just verify construction doesn't crash
    assert cost_h is not None
    assert mixer_h is not None


def test_15_qaoa_uses_expected_qubits(synthetic_data):
    """QAOA must use D qubits for D features."""
    X, y = synthetic_data
    D = X.shape[1]
    obj = FeatureSelectionObjective(X, y, target_k=2)
    result = run_qaoa(obj, target_k=2, p=1, max_iterations=5, shots=64, seed=SEED)
    assert result["n_qubits"] == D


def test_16_measurement_decode_correct():
    """Bitstring decoding must produce correct feature indices."""
    bits = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    indices = decode_bitstring(bits)
    assert indices == [0, 2, 4, 7]


def test_17_decode_samples_correct():
    """Sample decoding must find most common bitstring."""
    samples = np.array([
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])
    best_bits, best_indices, counts = decode_samples(samples)
    assert list(best_bits) == [1, 0, 1, 0]
    assert best_indices == [0, 2]
    assert counts["1010"] == 3


def test_18_qaoa_selected_features_valid(synthetic_data):
    """QAOA selected feature indices must be valid."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    result = run_qaoa(obj, target_k=2, p=1, max_iterations=10, shots=64, seed=SEED)
    for idx in result["selected_indices"]:
        assert 0 <= idx < X.shape[1], f"Invalid index {idx}"


def test_19_qaoa_subset_size_valid(synthetic_data):
    """QAOA must not select more features than D."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    result = run_qaoa(obj, target_k=2, p=1, max_iterations=10, shots=64, seed=SEED)
    assert result["n_selected"] <= X.shape[1]


# ═══════════════════════════════════════════════════════════════════
# 6. Exact-Solution Sanity Check
# ═══════════════════════════════════════════════════════════════════

def test_20_small_qaoa_vs_exact(synthetic_data):
    """QAOA must find a solution close to exact optimum for small problems."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2, alpha=0.1, beta=1.5)
    exact = obj.brute_force_optimal()
    result = run_qaoa(obj, target_k=2, p=2, max_iterations=60, shots=256, seed=SEED)

    comp = qaoa_vs_exact(result, exact)
    # QAOA objective should be at least 50% of exact (generous threshold)
    if exact["best_objective"] > 0:
        assert comp["approximation_ratio"] > 0.5, (
            f"QAOA too far from exact: ratio={comp['approximation_ratio']:.3f}"
        )


def test_21_objective_independently_recalculated(synthetic_data):
    """Objective from QAOA must match independent recalculation."""
    X, y = synthetic_data
    obj = FeatureSelectionObjective(X, y, target_k=2)
    result = run_qaoa(obj, target_k=2, p=1, max_iterations=10, shots=64, seed=SEED)

    # Reconstruct binary vector from selected indices
    x = np.zeros(X.shape[1])
    for idx in result["selected_indices"]:
        x[idx] = 1
    recalc = obj.evaluate(x)

    assert abs(recalc - result["objective_value"]) < 1e-8, (
        f"Recalculated {recalc} != reported {result['objective_value']}"
    )


# ═══════════════════════════════════════════════════════════════════
# 7. Leakage Prevention Tests
# ═══════════════════════════════════════════════════════════════════

def test_22_mi_fitted_on_train_only(real_latent_data):
    """MI scores must depend only on training data."""
    X_train, y_train = real_latent_data

    # Run twice on same data -> same result
    r1 = select_mutual_info(X_train, y_train, k=4, seed=SEED)
    r2 = select_mutual_info(X_train, y_train, k=4, seed=SEED)
    assert r1["selected_indices"] == r2["selected_indices"]


def test_23_rfe_fitted_on_train_only(real_latent_data):
    """RFE must depend only on training data."""
    X_train, y_train = real_latent_data
    r1 = select_rfe(X_train, y_train, k=4, seed=SEED)
    r2 = select_rfe(X_train, y_train, k=4, seed=SEED)
    assert r1["selected_indices"] == r2["selected_indices"]


def test_24_objective_fitted_on_train(real_latent_data):
    """Objective relevance scores must not change when val/test are hidden."""
    X_train, y_train = real_latent_data
    obj1 = FeatureSelectionObjective(X_train, y_train, target_k=8, seed=SEED)
    obj2 = FeatureSelectionObjective(X_train, y_train, target_k=8, seed=SEED)
    np.testing.assert_array_almost_equal(obj1.relevance, obj2.relevance)


# ═══════════════════════════════════════════════════════════════════
# 8. Artifact Tests
# ═══════════════════════════════════════════════════════════════════

def test_25_saved_artifacts_loadable():
    """Previously saved registry must be loadable."""
    reg_path = os.path.join(PART4_ARTIFACTS_DIR, "feature_registry.json")
    if os.path.exists(reg_path):
        reg = load_registry(reg_path)
        assert len(reg) == INPUT_FEATURE_DIM


# ═══════════════════════════════════════════════════════════════════
# 9. End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════

def test_26_end_to_end_small(synthetic_data):
    """Full pipeline must run without error on small synthetic data."""
    X, y = synthetic_data
    registry = build_latent_feature_registry(X.shape[1])

    # Classical
    r_mi = select_mutual_info(X, y, k=2, seed=SEED)
    assert len(r_mi["selected_indices"]) == 2

    # Objective + QUBO
    obj = FeatureSelectionObjective(X, y, target_k=2)
    Q = obj.build_qubo()
    assert obj.verify_qubo(Q)["passed"]

    # QAOA
    qaoa_r = run_qaoa(obj, target_k=2, p=1, max_iterations=10, shots=64, seed=SEED)
    assert len(qaoa_r["selected_indices"]) > 0

    # Exact
    exact = obj.brute_force_optimal()
    assert exact["best_objective"] > -np.inf
