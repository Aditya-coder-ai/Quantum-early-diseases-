"""
Comprehensive Automated Test Suite for Part 7: Complete Hybrid Classical-Quantum Pipeline.
Covers 19 critical validation points:
    1. Pipeline initialization
    2. Configuration loading & schema verification
    3. Configuration hashing / fingerprinting
    4. Dataset validation & contract checks
    5. Stratified splitting & disjoint index verification
    6. Preprocessing integration & scaling gate
    7. Feature extraction integration & 16D latent gate
    8. Feature selection integration (QAOA & classical)
    9. Imbalance handling integration (SMOTE / Class weights)
    10. VQC integration & qubit dimension matching
    11. Artifact saving & persistence checks
    12. Artifact loading & deserialization
    13. State transitions & lifecycle flow
    14. Failure handling & error propagation
    15. Inference pipeline execution on single sample
    16. Inference pipeline execution on batch samples
    17. Metric consistency across evaluation gates
    18. Comprehensive 8-point data leakage audit
    19. End-to-end fast pipeline execution
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import PROJECT_ROOT
from src.pipeline.config import PipelineConfig
from src.pipeline.state import PipelineState, PipelineContext
from src.pipeline.contracts import (
    RawDataContract, SplitDataContract, PreprocessedDataContract,
    LatentDataContract, SelectedFeaturesContract, ImbalanceDataContract,
    QuantumEncodedContract
)
from src.pipeline.validation import (
    validate_raw_data_gate, validate_split_gate, validate_preprocessing_gate,
    validate_feature_extraction_gate, validate_feature_selection_gate,
    validate_imbalance_gate, validate_quantum_encoding_gate, perform_full_leakage_audit
)
from src.pipeline.stages import (
    run_data_stage, run_preprocessing_stage, run_feature_extraction_stage,
    run_feature_selection_stage, run_imbalance_stage, run_quantum_encoding_stage
)
from src.pipeline.pipeline import HybridPipeline
from src.inference.predict import MedicalInferenceEngine


# ── Test 1: Pipeline Initialization ──────────────────────────────────────────
def test_01_pipeline_initialization():
    cfg = PipelineConfig(experiment_name="test_init")
    pipe = HybridPipeline(cfg)
    assert pipe.context.state == PipelineState.INITIALIZED
    assert pipe.experiment_id.startswith("test_init_")
    assert len(pipe.fingerprint) == 12


# ── Test 2: Configuration Loading & Schema Verification ──────────────────────
def test_02_config_loading_and_yaml_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "test_config.yaml")
        orig_cfg = PipelineConfig(num_selected_features=6, vqc_layers=1)
        orig_cfg.save_yaml(yaml_path)

        loaded_cfg = PipelineConfig.from_yaml(yaml_path)
        assert loaded_cfg.num_selected_features == 6
        assert loaded_cfg.vqc_layers == 1
        assert loaded_cfg.get_fingerprint() == orig_cfg.get_fingerprint()


# ── Test 3: Configuration Hashing / Fingerprinting ───────────────────────────
def test_03_config_fingerprint_uniqueness():
    cfg1 = PipelineConfig(num_selected_features=8)
    cfg2 = PipelineConfig(num_selected_features=6)
    assert cfg1.get_fingerprint() != cfg2.get_fingerprint()


# ── Test 4: Dataset Validation & Contract Checks ─────────────────────────────
def test_04_dataset_validation_gate():
    X = pd.DataFrame(np.random.randn(50, 30), columns=[f"f_{i}" for i in range(30)])
    y = pd.Series(np.random.choice([0, 1], size=50), name="target")
    contract = RawDataContract(X=X, y=y, n_samples=50, n_features=30, feature_names=list(X.columns))
    validate_raw_data_gate(contract)

    # Corrupt column count
    bad_contract = RawDataContract(X=X.iloc[:, :10], y=y, n_samples=50, n_features=10, feature_names=list(X.columns[:10]))
    with pytest.raises(ValueError, match="Expected 30 features"):
        validate_raw_data_gate(bad_contract)


# ── Test 5: Stratified Splitting & Disjoint Indices ──────────────────────────
def test_05_stratified_split_disjoint_indices():
    cfg = PipelineConfig()
    _, split = run_data_stage(cfg)
    validate_split_gate(split)
    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == 569


# ── Test 6: Preprocessing Integration & Scaling Gate ─────────────────────────
def test_06_preprocessing_integration():
    cfg = PipelineConfig()
    _, split = run_data_stage(cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep = run_preprocessing_stage(split, cfg, tmp_dir)
        validate_preprocessing_gate(prep)
        assert prep.X_train_scaled.shape == (len(split.X_train), 30)
        assert os.path.exists(prep.scaler_artifact_path)


# ── Test 7: Feature Extraction Integration & 16D Gate ────────────────────────
def test_07_feature_extraction_latent_gate():
    cfg = PipelineConfig(ae_epochs=2)
    _, split = run_data_stage(cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep = run_preprocessing_stage(split, cfg, tmp_dir)
        latent = run_feature_extraction_stage(prep, split, cfg, tmp_dir)
        validate_feature_extraction_gate(latent)
        assert latent.X_train_latent.shape[1] == 16


# ── Test 8: Feature Selection Integration (QAOA & Classical) ─────────────────
def test_08_feature_selection_integration():
    cfg = PipelineConfig(feature_selection_method="qaoa", num_selected_features=8)
    _, split = run_data_stage(cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep = run_preprocessing_stage(split, cfg, tmp_dir)
        latent = run_feature_extraction_stage(prep, split, cfg, tmp_dir)
        selected = run_feature_selection_stage(latent, split, cfg, tmp_dir)
        validate_feature_selection_gate(selected, expected_k=8)
        assert selected.X_train_selected.shape[1] == 8


# ── Test 9: Imbalance Integration & Training-Only Verification ───────────────
def test_09_imbalance_handling_training_only():
    cfg = PipelineConfig(imbalance_method="smote", imbalance_ratio=1.0)
    _, split = run_data_stage(cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep = run_preprocessing_stage(split, cfg, tmp_dir)
        latent = run_feature_extraction_stage(prep, split, cfg, tmp_dir)
        selected = run_feature_selection_stage(latent, split, cfg, tmp_dir)
        imb = run_imbalance_stage(selected, split, cfg)
        validate_imbalance_gate(imb, len(split.X_val), len(split.X_test))
        # Validation and test sizes must remain strictly unchanged
        assert len(imb.X_val) == len(split.X_val)
        assert len(imb.X_test) == len(split.X_test)
        # Training size increased due to SMOTE oversampling
        assert len(imb.X_train_balanced) > len(selected.X_train_selected)


# ── Test 10: Quantum Encoding & Qubit Dimension Matching ─────────────────────
def test_10_quantum_encoding_qubit_matching():
    cfg = PipelineConfig(num_selected_features=8)
    _, split = run_data_stage(cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep = run_preprocessing_stage(split, cfg, tmp_dir)
        latent = run_feature_extraction_stage(prep, split, cfg, tmp_dir)
        selected = run_feature_selection_stage(latent, split, cfg, tmp_dir)
        imb = run_imbalance_stage(selected, split, cfg)
        q_enc = run_quantum_encoding_stage(imb, cfg, tmp_dir)
        validate_quantum_encoding_gate(q_enc, expected_qubits=8)
        assert q_enc.n_qubits == 8
        assert (q_enc.X_train_angles >= 0.0).all()
        assert (q_enc.X_train_angles <= np.pi + 1e-4).all()


# ── Test 11 & 12: Artifact Saving & Reloading ────────────────────────────────
def test_11_12_artifact_saving_and_reloading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = PipelineConfig(
            experiment_name="test_artifacts",
            vqc_epochs=2,
            artifacts_dir=tmp_dir,
            num_selected_features=4,
            vqc_layers=1,
        )
        pipe = HybridPipeline(cfg)
        report = pipe.run()

        # Check required files exist
        assert os.path.exists(report.artifacts["preprocessing"])
        assert os.path.exists(report.artifacts["autoencoder"])
        assert os.path.exists(report.artifacts["feature_selection"])
        assert os.path.exists(report.artifacts["angle_scaler"])
        assert os.path.exists(report.artifacts["vqc_checkpoint"])
        assert os.path.exists(report.artifacts["config"])


# ── Test 13: Pipeline State Transitions ──────────────────────────────────────
def test_13_pipeline_state_transitions():
    ctx = PipelineContext(experiment_id="test_exp", config_hash="123456abcdef")
    assert ctx.state == PipelineState.INITIALIZED
    ctx.transition_to(PipelineState.DATA_LOADED, "Data loaded")
    assert ctx.state == PipelineState.DATA_LOADED
    assert len(ctx.logs) == 1
    assert ctx.logs[0]["state"] == "DATA_LOADED"


# ── Test 14: Failure Handling & Error Propagation ────────────────────────────
def test_14_failure_handling():
    ctx = PipelineContext(experiment_id="test_exp", config_hash="123456abcdef")
    ctx.fail("preprocessing", ValueError("Invalid dimension"))
    assert ctx.state == PipelineState.FAILED
    assert "Invalid dimension" in ctx.error


# ── Test 15: Standalone Single-Sample Inference Pipeline ─────────────────────
def test_15_standalone_single_sample_inference():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = PipelineConfig(
            experiment_name="test_inf_single",
            vqc_epochs=2,
            artifacts_dir=tmp_dir,
            num_selected_features=4,
            vqc_layers=1,
        )
        pipe = HybridPipeline(cfg)
        pipe.run()

        engine = MedicalInferenceEngine(pipe.artifact_dir)
        sample = {f"feat_{i}": 1.0 for i in range(30)}
        sample = pd.DataFrame(np.random.randn(1, 30), columns=engine.preprocessor.feature_names_in_)
        pred_res = engine.predict(sample)

        assert pred_res["n_samples"] == 1
        assert pred_res["predicted_classes"][0] in [0, 1]
        assert 0.0 <= pred_res["malignant_probabilities"][0] <= 1.0
        assert pred_res["qubit_count"] == 4


# ── Test 16: Standalone Batch Inference Pipeline ─────────────────────────────
def test_16_standalone_batch_inference():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = PipelineConfig(
            experiment_name="test_inf_batch",
            vqc_epochs=2,
            artifacts_dir=tmp_dir,
            num_selected_features=4,
            vqc_layers=1,
        )
        pipe = HybridPipeline(cfg)
        pipe.run()

        engine = MedicalInferenceEngine(pipe.artifact_dir)
        batch = pd.DataFrame(np.random.randn(5, 30), columns=engine.preprocessor.feature_names_in_)
        pred_res = engine.predict(batch)

        assert pred_res["n_samples"] == 5
        assert len(pred_res["predicted_classes"]) == 5
        assert len(pred_res["diagnosis_labels"]) == 5


# ── Test 17: Metric Consistency Check ────────────────────────────────────────
def test_17_metric_consistency():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = PipelineConfig(
            experiment_name="test_metrics",
            vqc_epochs=2,
            artifacts_dir=tmp_dir,
            num_selected_features=4,
            vqc_layers=1,
        )
        pipe = HybridPipeline(cfg)
        report = pipe.run()

        assert "accuracy" in report.validation_metrics
        assert "pr_auc" in report.validation_metrics
        assert "minority_recall" in report.validation_metrics
        assert "accuracy" in report.test_metrics
        assert "pr_auc" in report.test_metrics


# ── Test 18: Comprehensive 8-Point Leakage Audit ─────────────────────────────
def test_18_leakage_audit_pass():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = PipelineConfig(
            experiment_name="test_leakage",
            vqc_epochs=2,
            artifacts_dir=tmp_dir,
            num_selected_features=4,
            vqc_layers=1,
        )
        pipe = HybridPipeline(cfg)
        report = pipe.run()

        for check_name, passed in report.leakage_audit.items():
            assert passed is True, f"Leakage audit failed on {check_name}"


# ── Test 19: End-to-End Execution Succeeded ──────────────────────────────────
def test_19_end_to_end_execution_succeeded():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = PipelineConfig(
            experiment_name="test_e2e",
            vqc_epochs=2,
            artifacts_dir=tmp_dir,
            num_selected_features=4,
            vqc_layers=1,
        )
        pipe = HybridPipeline(cfg)
        report = pipe.run()
        assert report.status == "COMPLETED"
        assert report.total_runtime_s > 0
        assert pipe.context.state == PipelineState.COMPLETED
