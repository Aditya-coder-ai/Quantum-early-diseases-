"""
Comprehensive Automated Test Suite for Part 8: Model Explainability.
Covers 18 critical verification points:
    1. Modality router tabular detection
    2. Modality router CNN image detection
    3. Modality router rejects non-convolutional models for Grad-CAM
    4. VQC prediction wrapper shape and probability bounds
    5. Classical prediction wrapper probability output
    6. SHAP output dimensions and finite values
    7. Local explanation positive/negative driver separation
    8. Global feature importance ranking logic
    9. Permutation feature importance baseline
    10. Feature lineage tracker multi-level mapping
    11. QAOA vs. SHAP attribution comparison
    12. Grad-CAM convolutional layer hook and activation capture
    13. Grad-CAM heatmap [0, 1] normalization and finite bounds
    14. Faithfulness perturbation test
    15. Randomized model parameter sanity check
    16. Attribution stability across random seeds
    17. Zero data leakage verification
    18. Master explainer report generation and JSON/CSV serialization
"""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import PROJECT_ROOT
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.encoding import AngleScaler
from src.explainability.schemas import (
    LocalExplanation, GlobalFeatureImportance, FaithfulnessMetric,
    StabilityMetric, GradCAMExplanation, ExplanationReport
)
from src.explainability.router import DataModalityRouter, UnsupportedModalityError
from src.explainability.wrapper import VQCPredictionWrapper, ClassicalPredictionWrapper
from src.explainability.shap_explainer import HybridSHAPExplainer
from src.explainability.feature_importance import PermutationImportanceExplainer, extract_classical_model_importance
from src.explainability.feature_mapping import FeatureLineageTracker
from src.explainability.gradcam import GradCAMExplainer, create_synthetic_cnn
from src.explainability.validation import (
    evaluate_faithfulness, run_randomized_model_sanity_check,
    evaluate_attribution_stability, perform_explainability_leakage_audit
)
from src.explainability.explainer import ModelExplainer


# ── Test 1: Modality Router Tabular Detection ────────────────────────────────
def test_01_modality_router_tabular():
    X_df = pd.DataFrame(np.random.randn(10, 8), columns=[f"f_{i}" for i in range(8)])
    vqc = VariationalQuantumClassifier(n_qubits=8, n_layers=1, seed=42)
    decision = DataModalityRouter.route_explanation(X_df, vqc)
    assert decision["modality"] == "tabular"
    assert decision["strategy"] == "shap_kernel_and_permutation"
    assert decision["gradcam_supported"] is False


# ── Test 2: Modality Router Image with CNN ───────────────────────────────────
def test_02_modality_router_image_with_cnn():
    img_tensor = torch.randn(1, 1, 16, 16)
    cnn = create_synthetic_cnn()
    decision = DataModalityRouter.route_explanation(img_tensor, cnn)
    assert decision["modality"] == "image"
    assert decision["strategy"] == "gradcam_spatial_heatmap"
    assert decision["gradcam_supported"] is True
    assert decision["target_layer"] == "conv1"


# ── Test 3: Modality Router Rejects Tabular Models for Grad-CAM ──────────────
def test_03_modality_router_rejects_non_cnn():
    img_tensor = torch.randn(1, 1, 16, 16)
    linear_model = nn.Linear(16, 2)
    with pytest.raises(UnsupportedModalityError, match="contains no convolutional layers"):
        DataModalityRouter.route_explanation(img_tensor, linear_model, explicit_modality="image")


# ── Test 4: VQC Prediction Wrapper Output ────────────────────────────────────
def test_04_vqc_prediction_wrapper():
    vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    scaler = AngleScaler(target_range=(0.0, np.pi))
    X = np.random.randn(5, 4)
    scaler.fit(X)

    wrapper = VQCPredictionWrapper(vqc, angle_scaler=scaler, feature_names=[f"f_{i}" for i in range(4)])
    probs = wrapper.predict_proba(X)

    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert len(wrapper.predict_malignant_proba(X)) == 5


# ── Test 5: Classical Prediction Wrapper ─────────────────────────────────────
def test_05_classical_prediction_wrapper():
    clf = LogisticRegression()
    X = np.random.randn(20, 4)
    y = np.random.choice([0, 1], size=20)
    clf.fit(X, y)

    wrapper = ClassicalPredictionWrapper(clf, feature_names=[f"f_{i}" for i in range(4)])
    probs = wrapper.predict_proba(X)

    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ── Test 6: SHAP Output Dimensions and Finite Values ─────────────────────────
def test_06_shap_output_dimensions():
    vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    wrapper = VQCPredictionWrapper(vqc, feature_names=[f"f_{i}" for i in range(4)])
    X_bg = np.random.uniform(0, np.pi, size=(20, 4))
    X_eval = np.random.uniform(0, np.pi, size=(5, 4))

    explainer = HybridSHAPExplainer(
        predict_fn=wrapper.predict_malignant_proba,
        X_background=X_bg,
        background_size=15,
        seed=42
    )

    shap_mat, rankings = explainer.explain_dataset(X_eval, max_samples=5, nsamples=40)
    assert shap_mat.shape == (5, 4)
    assert not np.isnan(shap_mat).any()
    assert not np.isinf(shap_mat).any()
    assert len(rankings) == 4


# ── Test 7: Local Explanation Structure ──────────────────────────────────────
def test_07_local_explanation_structure():
    vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    wrapper = VQCPredictionWrapper(vqc, feature_names=[f"f_{i}" for i in range(4)])
    X_bg = np.random.uniform(0, np.pi, size=(20, 4))

    explainer = HybridSHAPExplainer(
        predict_fn=wrapper.predict_malignant_proba,
        X_background=X_bg,
        feature_names=[f"feat_{i}" for i in range(4)],
        background_size=15,
        seed=42
    )

    sample = np.random.uniform(0, np.pi, size=4)
    local_exp = explainer.explain_instance(sample, sample_id="TEST_01", nsamples=40)

    assert isinstance(local_exp, LocalExplanation)
    assert local_exp.sample_id == "TEST_01"
    assert local_exp.predicted_class in [0, 1]
    assert 0.0 <= local_exp.malignant_probability <= 1.0
    assert len(local_exp.shap_values) == 4
    assert len(local_exp.feature_values) == 4


# ── Test 8: Global Feature Importance Ranking Logic ──────────────────────────
def test_08_global_importance_ranking():
    vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    wrapper = VQCPredictionWrapper(vqc, feature_names=[f"f_{i}" for i in range(4)])
    X_bg = np.random.uniform(0, np.pi, size=(20, 4))
    X_eval = np.random.uniform(0, np.pi, size=(10, 4))

    explainer = HybridSHAPExplainer(
        predict_fn=wrapper.predict_malignant_proba,
        X_background=X_bg,
        background_size=15,
        seed=42
    )

    _, rankings = explainer.explain_dataset(X_eval, max_samples=10, nsamples=40)
    # Check descending order of mean_abs_shap
    for i in range(len(rankings) - 1):
        assert rankings[i].mean_abs_shap >= rankings[i + 1].mean_abs_shap
        assert rankings[i].rank == i + 1


# ── Test 9: Permutation Importance Baseline ──────────────────────────────────
def test_09_permutation_importance_baseline():
    def mock_predict(x):
        # Feature 0 dominates prediction
        return 1.0 / (1.0 + np.exp(-x[:, 0] * 3.0))

    X = np.random.randn(30, 4)
    y = (X[:, 0] > 0).astype(int)

    perm_exp = PermutationImportanceExplainer(predict_fn=mock_predict, metric="accuracy", n_repeats=3, seed=42)
    importance = perm_exp.compute_importance(X, y, feature_names=[f"f_{i}" for i in range(4)])

    assert len(importance) == 4
    # Feature 0 should have greatest importance
    assert importance["f_0"] >= importance["f_1"]


# ── Test 10: Feature Lineage Tracker Multi-Level Mapping ─────────────────────
def test_10_feature_lineage_tracker():
    raw_names = [f"raw_{i}" for i in range(30)]
    sel_indices = [0, 1, 3, 4, 6, 8, 9, 14]
    tracker = FeatureLineageTracker(raw_names, sel_indices, latent_dim=16)

    summary = tracker.get_selection_summary()
    assert len(summary) == 16
    assert summary["selected_by_qaoa"].sum() == 8

    influence_matrix = tracker.trace_raw_feature_importance_to_latent()
    assert influence_matrix.shape == (30, 16)


# ── Test 11: QAOA vs. SHAP Attribution Comparison ───────────────────────────
def test_11_qaoa_vs_shap_comparison():
    raw_names = [f"raw_{i}" for i in range(30)]
    sel_indices = [0, 1, 3, 4]
    tracker = FeatureLineageTracker(raw_names, sel_indices, latent_dim=16)

    shap_dict = {"latent_0": 0.15, "latent_1": 0.08, "latent_3": 0.22, "latent_4": 0.03}
    comp_df = tracker.build_qaoa_vs_shap_comparison(shap_dict)

    assert len(comp_df) == 4
    assert "shap_rank" in comp_df.columns
    assert "qaoa_selection_rank" in comp_df.columns


# ── Test 12 & 13: Grad-CAM Layer Detection & Heatmap Bounds ──────────────────
def test_12_13_gradcam_heatmap_generation():
    cnn = create_synthetic_cnn(in_channels=1, num_classes=2)
    gradcam = GradCAMExplainer(cnn, target_layer_name="conv1")

    dummy_img = torch.randn(1, 1, 8, 8)
    heatmap, explanation = gradcam.generate_heatmap(dummy_img, target_class=0)

    assert isinstance(explanation, GradCAMExplanation)
    assert heatmap.ndim == 2
    assert 0.0 <= explanation.heatmap_min <= explanation.heatmap_max <= 1.0
    assert not np.isnan(heatmap).any()


# ── Test 14: Faithfulness Perturbation Test ──────────────────────────────────
def test_14_faithfulness_perturbation():
    def mock_predict(x):
        # Feature 0 heavily increases probability
        arr = np.asarray(x)
        return np.clip(0.1 + 0.8 * arr[:, 0], 0.0, 1.0)

    x_sample = np.array([1.0, 0.2, 0.1, 0.05])
    shap_vals = [0.8, 0.1, 0.05, 0.01]
    feature_names = [f"f_{i}" for i in range(4)]

    fm = evaluate_faithfulness(
        predict_fn=mock_predict,
        x_sample=x_sample,
        shap_values=shap_vals,
        feature_names=feature_names,
        k=1,
        sample_id="P1"
    )

    assert isinstance(fm, FaithfulnessMetric)
    assert fm.is_faithful is True
    assert fm.top_k_delta > fm.least_k_delta


# ── Test 15: Randomized Model Parameter Sanity Check ─────────────────────────
def test_15_randomized_model_sanity():
    trained_shap = np.array([[0.5, 0.3, 0.1, 0.05]])

    # Completely inverted randomized function
    def rand_fn(x):
        arr = np.asarray(x)
        return 1.0 / (1.0 + np.exp(arr[:, 3] * 5.0))

    X_sample = np.array([[1.0, 1.0, 1.0, 1.0]])
    X_bg = np.random.uniform(0, 1, size=(10, 4))

    res = run_randomized_model_sanity_check(trained_shap, rand_fn, X_sample, X_bg)
    assert "passed" in res
    assert "pearson_correlation" in res


# ── Test 16: Attribution Stability Across Random Seeds ───────────────────────
def test_16_attribution_stability():
    def dummy_predict(x):
        arr = np.asarray(x)
        return 0.5 + 0.3 * np.tanh(arr[:, 0])

    X_sample = np.random.uniform(0, 1, size=(5, 3))
    X_bg = np.random.uniform(0, 1, size=(25, 3))
    feature_names = ["a", "b", "c"]

    stability_results = evaluate_attribution_stability(
        predict_fn=dummy_predict,
        X_sample=X_sample,
        X_train_bg=X_bg,
        feature_names=feature_names,
        n_runs=2,
        background_size=15,
    )

    assert len(stability_results) == 3
    assert all(isinstance(s, StabilityMetric) for s in stability_results)


# ── Test 17: Explainability Zero Data Leakage Verification ────────────────────
def test_17_leakage_audit():
    X_train_bg = np.random.randn(20, 4)
    X_val = np.random.randn(10, 4) + 50.0  # Completely disjoint values
    X_test = np.random.randn(10, 4) + 100.0

    shap_vals = np.random.randn(10, 4)

    audit = perform_explainability_leakage_audit(X_train_bg, X_val, X_test, shap_vals)
    assert audit["background_disjoint_from_val"] is True
    assert audit["background_disjoint_from_test"] is True
    assert audit["shap_values_finite"] is True
    assert audit["test_labels_unused_for_explanation"] is True


# ── Test 18: Master Explainer Report Serialization ───────────────────────────
def test_18_master_explainer_full_report():
    with tempfile.TemporaryDirectory() as tmp_dir:
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
        wrapper = VQCPredictionWrapper(vqc, feature_names=[f"f_{i}" for i in range(4)])

        X_bg = np.random.uniform(0, np.pi, size=(25, 4))
        X_val = np.random.uniform(0, np.pi, size=(10, 4))
        y_val = np.random.choice([0, 1], size=10)
        X_test = np.random.uniform(0, np.pi, size=(10, 4))

        explainer = ModelExplainer(
            vqc_wrapper=wrapper,
            X_train_bg=X_bg,
            feature_names=[f"f_{i}" for i in range(4)],
            artifacts_dir=tmp_dir,
            background_size=15,
            seed=42,
        )

        report = explainer.generate_full_report(
            X_val_cohort=X_val,
            y_val_cohort=y_val,
            X_test_cohort=X_test,
            experiment_id="test_exp_part8",
        )

        assert isinstance(report, ExplanationReport)
        assert os.path.exists(os.path.join(tmp_dir, "explainability_report.json"))
        assert os.path.exists(os.path.join(tmp_dir, "feature_importance.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "global_feature_importance_bar.png"))
