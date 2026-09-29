"""
Part 7: Modular Pipeline Stages.
Connects verified implementations from Parts 1 through 6 into standardized, contract-checked stages.
"""
from __future__ import annotations

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.config import PROJECT_ROOT, DATA_PROCESSED_DIR, MODELS_DIR
from src.data.loader import load_dataset
from src.preprocessing.pipeline import MedicalPreprocessingPipeline
from src.models.autoencoder import Autoencoder, train_autoencoder
from src.feature_selection.objective import FeatureSelectionObjective
from src.feature_selection.qaoa import run_qaoa
from src.feature_selection.classical import select_mutual_info, select_rfe
from src.imbalance.classical import compute_class_weights, smote_oversample
from src.imbalance.qgan import QuantumGAN
from src.vqc.encoding import AngleScaler, FeatureQubitMapper
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.training import train_vqc
from src.vqc.evaluation import evaluate_vqc_model, train_and_evaluate_classical_reference
from src.pipeline.contracts import (
    RawDataContract, SplitDataContract, PreprocessedDataContract,
    LatentDataContract, SelectedFeaturesContract, ImbalanceDataContract,
    QuantumEncodedContract, ModelPredictionContract
)
from src.pipeline.validation import (
    validate_raw_data_gate, validate_split_gate, validate_preprocessing_gate,
    validate_feature_extraction_gate, validate_feature_selection_gate,
    validate_imbalance_gate, validate_quantum_encoding_gate
)
from src.pipeline.config import PipelineConfig


# ── Stage 1: Data Loading & Stratified Splitting ─────────────────────────────
def run_data_stage(config: PipelineConfig) -> Tuple[RawDataContract, SplitDataContract]:
    """Loads raw dataset and creates stratified, leak-free train/val/test partitions."""
    X_raw, y_raw = load_dataset(save_raw=False)
    raw_contract = RawDataContract(
        X=X_raw,
        y=y_raw,
        n_samples=len(X_raw),
        n_features=X_raw.shape[1],
        feature_names=list(X_raw.columns),
    )
    validate_raw_data_gate(raw_contract)

    # 1. Stratified Train (70%) vs Temp (30%)
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X_raw, y_raw, test_size=(config.val_ratio + config.test_ratio),
        stratify=y_raw, random_state=config.seed
    )
    # 2. Stratified Val (15%) vs Test (15%)
    val_rel_ratio = config.val_ratio / (config.val_ratio + config.test_ratio)
    X_va, X_te, y_va, y_te = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_rel_ratio),
        stratify=y_temp, random_state=config.seed
    )

    split_contract = SplitDataContract(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
    )
    validate_split_gate(split_contract)
    return raw_contract, split_contract


# ── Stage 2: Leak-Free Preprocessing ─────────────────────────────────────────
def run_preprocessing_stage(
    split: SplitDataContract,
    config: PipelineConfig,
    artifact_dir: str
) -> PreprocessedDataContract:
    """Fits preprocessor exclusively on training split; transforms val and test splits."""
    pipeline = MedicalPreprocessingPipeline(scaling_method=config.scaling_method)
    pipeline.fit(split.X_train)

    scaler_path = os.path.join(artifact_dir, "preprocessing_pipeline.joblib")
    pipeline.save(scaler_path)

    X_tr_scaled = np.asarray(pipeline.transform(split.X_train))
    X_va_scaled = np.asarray(pipeline.transform(split.X_val))
    X_te_scaled = np.asarray(pipeline.transform(split.X_test))

    prep_contract = PreprocessedDataContract(
        X_train_scaled=X_tr_scaled,
        X_val_scaled=X_va_scaled,
        X_test_scaled=X_te_scaled,
        feature_names=split.X_train.columns.tolist(),
        scaler_artifact_path=scaler_path,
    )
    validate_preprocessing_gate(prep_contract)
    return prep_contract


# ── Stage 3: Classical Autoencoder Representation ────────────────────────────
def run_feature_extraction_stage(
    prep: PreprocessedDataContract,
    split: SplitDataContract,
    config: PipelineConfig,
    artifact_dir: str
) -> LatentDataContract:
    """Compresses 30 continuous clinical features into 16 latent dimensions."""
    ae_model_path = os.path.join(artifact_dir, "autoencoder.pth")
    input_dim = prep.X_train_scaled.shape[1]

    autoencoder = Autoencoder(
        input_dim=input_dim,
        hidden_dims=config.ae_hidden_dims,
        latent_dim=config.latent_dim,
    )

    # Check if pre-existing trained weights exist to save computation
    existing_ae = os.path.join(MODELS_DIR, "autoencoder.pth")
    if os.path.exists(existing_ae):
        ckpt = torch.load(existing_ae, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
    else:
        X_tr_tensor = torch.tensor(prep.X_train_scaled, dtype=torch.float32)
        X_va_tensor = torch.tensor(prep.X_val_scaled, dtype=torch.float32)
        train_autoencoder(
            autoencoder, X_tr_tensor, X_va_tensor,
            epochs=config.ae_epochs, batch_size=config.ae_batch_size, lr=config.ae_lr
        )
    
    # Save checkpoint to experiment artifact dir
    torch.save({
        "model_state_dict": autoencoder.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": config.ae_hidden_dims,
        "latent_dim": config.latent_dim,
    }, ae_model_path)
    autoencoder.eval()

    with torch.no_grad():
        z_tr = autoencoder.encode(torch.tensor(prep.X_train_scaled, dtype=torch.float32)).numpy()
        z_va = autoencoder.encode(torch.tensor(prep.X_val_scaled, dtype=torch.float32)).numpy()
        z_te = autoencoder.encode(torch.tensor(prep.X_test_scaled, dtype=torch.float32)).numpy()

    latent_contract = LatentDataContract(
        X_train_latent=z_tr,
        X_val_latent=z_va,
        X_test_latent=z_te,
        latent_dim=config.latent_dim,
        model_artifact_path=ae_model_path,
    )
    validate_feature_extraction_gate(latent_contract)
    return latent_contract


# ── Stage 4: Feature Selection (QAOA vs. Classical) ──────────────────────────
def run_feature_selection_stage(
    latent: LatentDataContract,
    split: SplitDataContract,
    config: PipelineConfig,
    artifact_dir: str
) -> SelectedFeaturesContract:
    """Selects an optimal compact feature subset (K=8) using QAOA or classical selectors."""
    k = config.num_selected_features
    selector_artifact_path = os.path.join(artifact_dir, "feature_selection.json")

    if config.feature_selection_method == "qaoa":
        # Check pre-computed Part 4 QAOA artifact for deterministic reuse
        qaoa_json = os.path.join(PROJECT_ROOT, "features", "selected", "qaoa", "qaoa_selection_primary.json")
        if os.path.exists(qaoa_json):
            with open(qaoa_json, "r") as f:
                qaoa_meta = json.load(f)
            indices = qaoa_meta["selected_indices"]
            if len(indices) >= k:
                selected_indices = indices[:k]
            else:
                remaining = [i for i in range(latent.latent_dim) if i not in indices]
                selected_indices = indices + remaining[:k - len(indices)]
        else:
            obj = FeatureSelectionObjective(
                latent.X_train_latent, split.y_train.values,
                target_k=k, seed=config.seed
            )
            qaoa_res = run_qaoa(obj, target_k=k, p=2, max_iterations=20, shots=1024, seed=config.seed)
            selected_indices = qaoa_res["selected_indices"]
    elif config.feature_selection_method == "mutual_info":
        res = select_mutual_info(latent.X_train_latent, split.y_train.values, k=k, seed=config.seed)
        selected_indices = res["selected_indices"]
    elif config.feature_selection_method == "rfe":
        res = select_rfe(latent.X_train_latent, split.y_train.values, k=k, seed=config.seed)
        selected_indices = res["selected_indices"]
    elif config.feature_selection_method == "none":
        selected_indices = list(range(k))
    else:
        raise ValueError(f"Unknown feature_selection_method: {config.feature_selection_method}")

    selected_names = [f"latent_{i}" for i in selected_indices]
    X_tr_sel = latent.X_train_latent[:, selected_indices]
    X_va_sel = latent.X_val_latent[:, selected_indices]
    X_te_sel = latent.X_test_latent[:, selected_indices]

    # Save selector metadata
    sel_meta = {
        "method": config.feature_selection_method,
        "k": k,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
    }
    with open(selector_artifact_path, "w") as f:
        json.dump(sel_meta, f, indent=2)

    selected_contract = SelectedFeaturesContract(
        X_train_selected=X_tr_sel,
        X_val_selected=X_va_sel,
        X_test_selected=X_te_sel,
        selected_indices=selected_indices,
        selected_names=selected_names,
        method=config.feature_selection_method,
        k=k,
        artifact_path=selector_artifact_path,
    )
    validate_feature_selection_gate(selected_contract, expected_k=k)
    return selected_contract


# ── Stage 5: Class Imbalance Handling ────────────────────────────────────────
def run_imbalance_stage(
    selected: SelectedFeaturesContract,
    split: SplitDataContract,
    config: PipelineConfig
) -> ImbalanceDataContract:
    """Applies oversampling (SMOTE/QGAN) or computes class-weighted loss parameters strictly on train."""
    method = config.imbalance_method
    X_tr = selected.X_train_selected
    y_tr = split.y_train.values

    class_weights = None

    if method == "smote":
        X_tr_bal, y_tr_bal, _ = smote_oversample(
            X_tr, y_tr, ratio=config.imbalance_ratio, seed=config.seed
        )
    elif method == "class_weight":
        class_weights = compute_class_weights(y_tr)
        X_tr_bal, y_tr_bal = X_tr, y_tr
    elif method == "qgan":
        deficit = int(round((np.sum(y_tr == 1) - np.sum(y_tr == 0)) * config.imbalance_ratio))
        X_min = X_tr[y_tr == 0]
        qgan = QuantumGAN(n_qubits=selected.k, n_layers=2, seed=config.seed)
        qgan.train(X_min, epochs=25, batch_size=16, verbose=False)
        X_synth = qgan.generate(deficit, seed=config.seed)
        X_tr_bal = np.vstack([X_tr, X_synth])
        y_tr_bal = np.hstack([y_tr, np.full(deficit, 0, dtype=y_tr.dtype)])
    elif method == "none":
        X_tr_bal, y_tr_bal = X_tr, y_tr
    else:
        raise ValueError(f"Unknown imbalance_method: {method}")

    imb_contract = ImbalanceDataContract(
        X_train_balanced=X_tr_bal,
        y_train_balanced=y_tr_bal,
        X_val=selected.X_val_selected,
        y_val=split.y_val.values,
        X_test=selected.X_test_selected,
        y_test=split.y_test.values,
        method=method,
        class_weights=class_weights,
    )
    validate_imbalance_gate(imb_contract, len(split.X_val), len(split.X_test))
    return imb_contract


# ── Stage 6: Quantum Feature Encoding ────────────────────────────────────────
def run_quantum_encoding_stage(
    imb: ImbalanceDataContract,
    config: PipelineConfig,
    artifact_dir: str
) -> QuantumEncodedContract:
    """Normalizes features to [0, pi] rotation angles using training-fitted AngleScaler."""
    scaler = AngleScaler(target_range=(0.0, np.pi))
    scaler.fit(imb.X_train_balanced)

    angle_scaler_path = os.path.join(artifact_dir, "angle_scaler.json")
    scaler.save(angle_scaler_path)

    X_tr_ang = scaler.transform(imb.X_train_balanced)
    X_va_ang = scaler.transform(imb.X_val)
    X_te_ang = scaler.transform(imb.X_test)

    q_contract = QuantumEncodedContract(
        X_train_angles=X_tr_ang,
        X_val_angles=X_va_ang,
        X_test_angles=X_te_ang,
        angle_scaler_path=angle_scaler_path,
        n_qubits=imb.X_train_balanced.shape[1],
    )
    validate_quantum_encoding_gate(q_contract, expected_qubits=config.num_selected_features)
    return q_contract


# ── Stage 7: VQC Training & Validation ───────────────────────────────────────
def run_vqc_training_stage(
    q_enc: QuantumEncodedContract,
    imb: ImbalanceDataContract,
    config: PipelineConfig,
    artifact_dir: str
) -> Tuple[VariationalQuantumClassifier, ModelPredictionContract, Dict[str, Any]]:
    """Initializes and trains the Variational Quantum Classifier with early stopping."""
    model = VariationalQuantumClassifier(
        n_qubits=q_enc.n_qubits,
        n_layers=config.vqc_layers,
        ansatz_type=config.vqc_ansatz,
        entanglement=config.vqc_entanglement,
        preferred_device=config.preferred_device,
        fallback_device=config.fallback_device,
        diff_method=config.diff_method,
        seed=config.seed,
    )

    train_res = train_vqc(
        model,
        X_train=q_enc.X_train_angles,
        y_train=imb.y_train_balanced,
        X_val=q_enc.X_val_angles,
        y_val=imb.y_val,
        epochs=config.vqc_epochs,
        batch_size=config.vqc_batch_size,
        lr=config.vqc_lr,
        class_weights=imb.class_weights,
        early_stopping_patience=config.vqc_patience,
        verbose=False,
    )

    vqc_model_path = os.path.join(artifact_dir, "vqc_model.pt")
    model.save(vqc_model_path)

    # Validation evaluation
    val_metrics = evaluate_vqc_model(model, q_enc.X_val_angles, imb.y_val, minority_class=0)
    val_probs = model.predict_proba(q_enc.X_val_angles)
    val_preds = (val_probs >= 0.5).astype(int)

    val_contract = ModelPredictionContract(
        y_pred=val_preds,
        y_probs=val_probs,
        metrics=val_metrics,
        inference_time_s=val_metrics["inference_time_s"],
        model_name="VQC",
    )
    return model, val_contract, train_res


# ── Stage 8: Final Single-Pass Test Evaluation & Reporting ───────────────────
def run_test_evaluation_stage(
    model: VariationalQuantumClassifier,
    q_enc: QuantumEncodedContract,
    imb: ImbalanceDataContract,
    config: PipelineConfig
) -> Tuple[ModelPredictionContract, Dict[str, Any]]:
    """Evaluates the final candidate VQC model on the held-out test split exactly once."""
    test_metrics = evaluate_vqc_model(model, q_enc.X_test_angles, imb.y_test, minority_class=0)
    test_probs = model.predict_proba(q_enc.X_test_angles)
    test_preds = (test_probs >= 0.5).astype(int)

    test_contract = ModelPredictionContract(
        y_pred=test_preds,
        y_probs=test_probs,
        metrics=test_metrics,
        inference_time_s=test_metrics["inference_time_s"],
        model_name="VQC_Test",
    )

    # Also evaluate Classical Reference Model on test set for direct fair benchmark
    classical_test = train_and_evaluate_classical_reference(
        config.classical_baseline,
        imb.X_train_balanced,
        imb.y_train_balanced,
        imb.X_test,
        imb.y_test,
        class_weight="balanced" if config.imbalance_method == "class_weight" else None,
        seed=config.seed,
        minority_class=0,
    )

    return test_contract, classical_test
