"""
Automated Test Suite for Part 9: Classical vs Hybrid Comparison.
Validates split integrity, metric computation, threshold tuning, fairness auditor,
paired statistical significance tests, ablations, plotting, and report generation.
"""
from __future__ import annotations

import os
import shutil
import pytest
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from configs.config import PROJECT_ROOT
from src.comparison.config import ComparisonConfig, ModelConfig
from src.comparison.metrics import (
    compute_extended_metrics, tune_decision_threshold,
    compute_calibration_curve_data, ResourceProfiler
)
from src.comparison.fairness import FairnessAuditor, FairnessViolationError
from src.comparison.statistical import aggregate_multiseed_metrics, run_paired_statistical_tests
from src.comparison.benchmark import ModelBenchmarkEngine
from src.comparison.ablations import AblationSuite
from src.comparison.visualization import ComparisonVisualizer
from src.comparison.reporter import ComparisonReporter
from src.comparison.runner import ComparisonRunner


@pytest.fixture
def quick_config(tmp_path):
    """Creates a lightweight test configuration."""
    cfg = ComparisonConfig(
        experiment_name="test_comparison",
        seeds=[42],
        primary_seed=42,
        artifacts_dir=str(tmp_path / "artifacts"),
        vqc_epochs=2,
        vqc_patience=1,
        quick_mode=True,
        ablation_feature_counts=[4, 8],
        ablation_circuit_depths=[1, 2],
        ablation_data_fractions=[0.5, 1.0],
        robustness_noise_sigmas=[0.0, 0.05],
        quantum_noise_shots=[1024],
        models=[
            ModelConfig(
                model_id="logreg_test",
                model_name="Test Logistic Regression",
                model_family="classical",
                input_space="raw_30",
                n_features=30,
                hyperparameters={"max_iter": 50}
            ),
            ModelConfig(
                model_id="compact_svm_test",
                model_name="Test Compact SVM",
                model_family="compact_classical",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                n_features=8,
                hyperparameters={"C": 1.0, "kernel": "rbf", "probability": True}
            ),
            ModelConfig(
                model_id="vqc_test",
                model_name="Test Hybrid VQC",
                model_family="hybrid_vqc",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                imbalance_method="none",
                n_features=8,
                n_qubits=8,
                circuit_depth=1
            )
        ]
    )
    return cfg


# ── Test 1: Common Split Loading & Dimensionality ────────────────────────────
def test_01_common_split_loading_and_shape(quick_config):
    engine = ModelBenchmarkEngine(quick_config)
    assert engine.splits["X_train_raw"].shape == (398, 30)
    assert engine.splits["X_val_raw"].shape == (85, 30)
    assert engine.splits["X_test_raw"].shape == (86, 30)
    assert len(engine.splits["y_test"]) == 86
    assert engine.latent_splits["X_train_latent"].shape == (398, 16)
    assert engine.qaoa_selected_splits["X_train_qaoa"].shape == (398, 8)


# ── Test 2: Model Configuration & YAML Serialization ─────────────────────────
def test_02_model_configuration_and_yaml(tmp_path):
    yaml_file = tmp_path / "test_cfg.yaml"
    cfg = ComparisonConfig(experiment_name="yaml_test", seeds=[42, 43])
    cfg.to_yaml(str(yaml_file))
    assert os.path.exists(yaml_file)

    loaded = ComparisonConfig.from_yaml(str(yaml_file))
    assert loaded.experiment_name == "yaml_test"
    assert loaded.seeds == [42, 43]
    assert len(loaded.models) > 0


# ── Test 3: Metric Calculation Correctness ───────────────────────────────────
def test_03_extended_metrics_calculation():
    # 0 = Malignant (disease), 1 = Benign
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])  # 3 TP, 1 FN, 3 TN, 1 FP for disease
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.85, 0.7, 0.4])

    m = compute_extended_metrics(y_true, y_pred, y_prob, disease_class=0)
    assert m["tp"] == 3
    assert m["fn"] == 1
    assert m["tn"] == 3
    assert m["fp"] == 1
    assert m["recall"] == 0.75
    assert m["specificity"] == 0.75
    assert m["precision"] == 0.75
    assert m["accuracy"] == 0.75
    assert m["false_negatives"] == 1
    assert m["roc_auc"] is not None
    assert m["pr_auc"] is not None


# ── Test 4: Confusion Matrix & False Negatives ───────────────────────────────
def test_04_confusion_matrix_and_false_negatives():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])  # Perfect recall: 0 FN
    m_perfect = compute_extended_metrics(y_true, y_pred, disease_class=0)
    assert m_perfect["false_negatives"] == 0
    assert m_perfect["fnr"] == 0.0

    y_pred_bad = np.array([1, 1, 1, 1])  # 2 FN
    m_bad = compute_extended_metrics(y_true, y_pred_bad, disease_class=0)
    assert m_bad["false_negatives"] == 2
    assert m_bad["fnr"] == 1.0


# ── Test 5: ROC-AUC and PR-AUC Calculation ───────────────────────────────────
def test_05_roc_auc_and_pr_auc():
    y_true = np.array([0, 0, 1, 1])
    # Prob of benign: lower means predicted malignant
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_extended_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob, disease_class=0)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


# ── Test 6: Threshold Tuning Policy ──────────────────────────────────────────
def test_06_threshold_tuning_policy():
    y_val = np.array([0, 0, 0, 1, 1, 1, 1])
    y_val_probs = np.array([0.15, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85])

    best_th, best_m, sweep = tune_decision_threshold(
        y_val=y_val,
        y_val_probs=y_val_probs,
        metric="f1",
        disease_class=0
    )
    assert 0.1 <= best_th <= 0.9
    assert len(sweep) > 0
    assert "f1" in best_m


# ── Test 7: Calibration Curve Coordinates ────────────────────────────────────
def test_07_calibration_curve_generation():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.85])
    cal = compute_calibration_curve_data(y_true, y_prob, disease_class=0, n_bins=4)
    assert "prob_true" in cal and "prob_pred" in cal
    assert len(cal["prob_true"]) == len(cal["prob_pred"])
    assert all(0.0 <= p <= 1.0 for p in cal["prob_true"])


# ── Test 8: Fairness Auditor Pass and Rejection ──────────────────────────────
def test_08_fairness_auditor_pass_and_rejection():
    canonical_x = np.ones((86, 30))
    canonical_y = np.zeros(86)
    auditor = FairnessAuditor(canonical_x, canonical_y)

    # Valid check
    assert auditor.audit_test_split("model_ok", canonical_x, canonical_y) is True

    # Modified length check
    with pytest.raises(FairnessViolationError):
        auditor.audit_test_split("model_bad", canonical_x[:50], canonical_y[:50])

    # Modified label check
    with pytest.raises(FairnessViolationError):
        tampered_y = canonical_y.copy()
        tampered_y[0] = 1
        auditor.audit_test_split("model_tampered", canonical_x, tampered_y)


# ── Test 9: Imbalance Isolation Check ────────────────────────────────────────
def test_09_imbalance_isolation_check():
    canonical_x = np.ones((86, 30))
    canonical_y = np.zeros(86)
    auditor = FairnessAuditor(canonical_x, canonical_y)

    assert auditor.audit_imbalance_isolation("model_clean", val_len=85, test_len=86) is True

    # Leaked validation set (e.g., SMOTE applied to validation)
    with pytest.raises(FairnessViolationError):
        auditor.audit_imbalance_isolation("model_leak", val_len=120, test_len=86)


# ── Test 10: Multi-Seed Aggregation ──────────────────────────────────────────
def test_10_multiseed_aggregation():
    runs = [
        {"model_id": "svm", "model_name": "SVM", "accuracy": 0.95, "recall": 0.96, "specificity": 0.94,
         "precision": 0.95, "f1": 0.955, "roc_auc": 0.99, "pr_auc": 0.98, "false_negatives": 1,
         "training_time_s": 0.1, "inference_latency_ms": 0.02},
        {"model_id": "svm", "model_name": "SVM", "accuracy": 0.97, "recall": 0.98, "specificity": 0.96,
         "precision": 0.97, "f1": 0.975, "roc_auc": 0.995, "pr_auc": 0.99, "false_negatives": 0,
         "training_time_s": 0.12, "inference_latency_ms": 0.022}
    ]
    summary = aggregate_multiseed_metrics(runs)
    assert "svm" in summary
    assert summary["svm"]["n_seeds"] == 2
    assert summary["svm"]["metrics"]["recall"]["mean"] == 0.97
    assert summary["svm"]["metrics"]["false_negatives"]["mean"] == 0.5


# ── Test 11: Paired Statistical Tests ────────────────────────────────────────
def test_11_paired_statistical_tests():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Baseline: 1 mistake
    base_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.75])
    base_preds = (base_probs >= 0.5).astype(int)
    # Hybrid: 0 mistakes
    hyb_probs = np.array([0.05, 0.1, 0.15, 0.95, 0.9, 0.85])
    hyb_preds = (hyb_probs >= 0.5).astype(int)

    res = run_paired_statistical_tests(y_true, base_probs, base_preds, hyb_probs, hyb_preds)
    assert "paired_t_test" in res
    assert "wilcoxon_test" in res
    assert "mcnemar_test" in res
    assert "p_value" in res["paired_t_test"]
    assert "contingency_table" in res["mcnemar_test"]


# ── Test 12: Explainability Comparison ───────────────────────────────────────
def test_12_explainability_comparison(quick_config):
    engine = ModelBenchmarkEngine(quick_config)
    X_tr = engine.qaoa_selected_splits["X_train_qaoa"]
    X_te = engine.qaoa_selected_splits["X_test_qaoa"]
    y_tr = engine.splits["y_train"]

    svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=42)
    svm.fit(X_tr, y_tr)

    from src.vqc.encoding import AngleScaler
    from src.vqc.model import VariationalQuantumClassifier
    scaler = AngleScaler(target_range=(0.0, np.pi)).fit(X_tr)
    vqc = VariationalQuantumClassifier(n_qubits=8, n_layers=1, seed=42)

    from src.comparison.explainability_comparison import compare_classical_vs_hybrid_explanations
    feature_names = [f"feat_{i}" for i in range(8)]
    exp_res = compare_classical_vs_hybrid_explanations(
        classical_model=svm,
        hybrid_vqc=vqc,
        angle_scaler=scaler,
        X_train=X_tr,
        X_test=X_te,
        feature_names=feature_names,
        n_background=10,
        n_explain_samples=5,
        seed=42
    )

    assert "spearman_rank_correlation" in exp_res
    assert len(exp_res["comparison_table"]) == 8
    assert "top_classical_feature" in exp_res
    assert "top_vqc_feature" in exp_res


# ── Test 13: Controlled Matrix Ablations A–E ─────────────────────────────────
def test_13_ablation_matrix_execution(quick_config):
    engine = ModelBenchmarkEngine(quick_config)
    suite = AblationSuite(engine)
    matrix_res = suite.run_ablation_matrix(seed=42)
    assert len(matrix_res) == 5
    codes = [r["ablation_code"] for r in matrix_res]
    assert "Ablation_A_ClassicalFS_Classical" in codes
    assert "Ablation_C_QAOAFS_VQC_Original" in codes
    assert "Ablation_D_QAOAFS_SMOTE_VQC" in codes


# ── Test 14: Feature Count and Depth Ablations ───────────────────────────────
def test_14_feature_count_and_depth_ablations(quick_config):
    engine = ModelBenchmarkEngine(quick_config)
    suite = AblationSuite(engine)
    f_res = suite.run_feature_count_ablation(seed=42)
    assert len(f_res) > 0
    assert any(x["model_type"] == "Classical_SVM" for x in f_res)
    assert any(x["model_type"] == "Hybrid_VQC" for x in f_res)

    d_res = suite.run_circuit_depth_ablation(seed=42)
    assert len(d_res) > 0
    assert d_res[0]["circuit_depth"] in [1, 2, 3]


# ── Test 15: Noise and Perturbation Robustness ───────────────────────────────
def test_15_noise_and_perturbation_robustness(quick_config):
    engine = ModelBenchmarkEngine(quick_config)
    suite = AblationSuite(engine)
    rob_res = suite.run_robustness_experiments(seed=42)
    assert "feature_perturbation" in rob_res
    assert "quantum_simulation_noise" in rob_res
    assert len(rob_res["feature_perturbation"]) > 0


# ── Test 16: Visualization Suite Rendering ───────────────────────────────────
def test_16_visualization_suite(quick_config, tmp_path):
    plots_dir = tmp_path / "plots"
    vis = ComparisonVisualizer(str(plots_dir))

    fake_bench = [{
        "model_id": "svm", "model_name": "SVM", "recall": 0.96, "specificity": 0.94,
        "f1": 0.95, "roc_auc": 0.99, "pr_auc": 0.98, "false_negatives": 1,
        "tn": 51, "fp": 3, "fn": 1, "tp": 31, "test_probs": [0.1]*32 + [0.9]*54,
        "training_time_s": 0.1, "inference_latency_ms": 0.05, "feature_reduction_pct": 0.0,
        "calibration_curve": {"prob_true": [0.0, 1.0], "prob_pred": [0.1, 0.9]},
    }]
    fake_stats = {"paired_tests": {}, "multiseed_summary": {}}
    fake_abl = {
        "feature_count_ablation": [{"model_type": "Classical_SVM", "k_features": 8, "recall": 0.96, "pr_auc": 0.98}],
        "circuit_depth_ablation": [{"circuit_depth": 2, "recall": 0.96, "pr_auc": 0.98, "trainable_parameters": 34}],
        "robustness": {"feature_perturbation": [{"noise_sigma": 0.0, "svm_recall": 0.96, "vqc_recall": 0.96}]}
    }
    fake_exp = {
        "spearman_rank_correlation": 0.85,
        "comparison_table": [{"feature_name": "latent_0", "classical_mean_abs_shap": 0.1, "vqc_mean_abs_shap": 0.12}]
    }

    y_test = np.array([0]*32 + [1]*54)
    saved = vis.generate_all_plots(fake_bench, fake_stats, fake_abl, fake_exp, y_test)
    assert len(saved) == 13
    for p in saved:
        assert os.path.exists(p)


# ── Test 17: Report Generation and Serialization ─────────────────────────────
def test_17_report_generation_and_serialization(quick_config, tmp_path):
    reporter = ComparisonReporter(str(tmp_path / "artifacts"))
    fake_bench = [{
        "model_id": "svm", "model_name": "SVM", "model_family": "classical",
        "input_space": "raw_30", "n_features": 30, "feature_reduction_pct": 0.0,
        "threshold": 0.5, "recall": 0.96, "specificity": 0.94, "precision": 0.95,
        "f1": 0.95, "roc_auc": 0.99, "pr_auc": 0.98, "mcc": 0.90, "balanced_accuracy": 0.95,
        "false_negatives": 1, "brier_score": 0.04, "training_time_s": 0.1,
        "inference_latency_ms": 0.05, "n_qubits": 0, "circuit_depth": 0,
        "quantum_gates": 0, "trainable_parameters": 240, "backend": "CPU",
        "tn": 51, "fp": 3, "fn": 1, "tp": 31,
    }]
    fake_stats = {"paired_t_test": {"t_statistic": 1.2, "p_value": 0.23, "statistically_significant_p05": False}}
    fake_multi = {"svm": {"model_name": "SVM", "n_seeds": 1, "metrics": {"recall": {"formatted": "0.9600 ± 0.0000"}}}}
    fake_abl = {"ablation_matrix": []}
    fake_exp = {"spearman_rank_correlation": 0.8, "top_classical_feature": "latent_8", "top_vqc_feature": "latent_8"}
    fake_fair = {"total_checks": 5, "all_passed": True}

    rep_path = reporter.generate_and_save(fake_bench, fake_multi, fake_stats, fake_abl, fake_exp, fake_fair, [])
    assert os.path.exists(rep_path)
    assert os.path.exists(tmp_path / "artifacts" / "results" / "results.csv")
    assert os.path.exists(tmp_path / "artifacts" / "results" / "results.json")
    assert os.path.exists(tmp_path / "artifacts" / "results" / "statistical_tests.json")


# ── Test 18: End-to-End Runner Quick Mode ────────────────────────────────────
def test_18_end_to_end_runner_quick_mode(quick_config):
    runner = ComparisonRunner(quick_config)
    res = runner.run()
    assert res["status"] == "COMPLETE"
    assert os.path.exists(res["report_path"])
    assert len(res["benchmark_results"]) == len(quick_config.models)
    assert len(res["plot_paths"]) == 13
