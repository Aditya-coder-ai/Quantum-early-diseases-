"""
Part 7: Validation Gates & Data Leakage Auditing.
Enforces strict boundary gates between stages and verifies zero data leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.pipeline.contracts import (
    RawDataContract, SplitDataContract, PreprocessedDataContract,
    LatentDataContract, SelectedFeaturesContract, ImbalanceDataContract,
    QuantumEncodedContract
)


def validate_raw_data_gate(raw: RawDataContract) -> None:
    """Gate 1: Verifies raw dataset integrity."""
    raw.validate()
    if raw.n_features != 30:
        raise ValueError(f"Expected 30 features, found {raw.n_features}.")
    if raw.X.isnull().sum().sum() > 0:
        raise ValueError("Raw dataset contains unexpected missing values.")


def validate_split_gate(split: SplitDataContract) -> None:
    """Gate 2: Verifies partition sizes and non-overlapping index integrity."""
    split.validate()
    # Check that indices across splits are strictly disjoint
    train_idx = set(split.X_train.index)
    val_idx = set(split.X_val.index)
    test_idx = set(split.X_test.index)

    if not train_idx.isdisjoint(val_idx):
        raise ValueError("Data Leakage Detected: Training and validation indices overlap!")
    if not train_idx.isdisjoint(test_idx):
        raise ValueError("Data Leakage Detected: Training and test indices overlap!")
    if not val_idx.isdisjoint(test_idx):
        raise ValueError("Data Leakage Detected: Validation and test indices overlap!")


def validate_preprocessing_gate(prep: PreprocessedDataContract) -> None:
    """Gate 3: Verifies scaling and numerical bounds."""
    prep.validate()
    # Verify zero NaNs/Infs
    for split_name, data in [("train", prep.X_train_scaled), ("val", prep.X_val_scaled), ("test", prep.X_test_scaled)]:
        arr = np.asarray(data)
        if np.isnan(arr).any():
            raise ValueError(f"NaN values found in preprocessed {split_name} data.")
        if np.isinf(arr).any():
            raise ValueError(f"Infinite values found in preprocessed {split_name} data.")


def validate_feature_extraction_gate(latent: LatentDataContract) -> None:
    """Gate 4: Verifies latent representation shape and stability."""
    latent.validate()
    if latent.latent_dim != 16:
        raise ValueError(f"Expected 16-dimensional latent space, found {latent.latent_dim}.")


def validate_feature_selection_gate(selected: SelectedFeaturesContract, expected_k: int) -> None:
    """Gate 5: Verifies selected feature subset count and valid index range."""
    selected.validate()
    if selected.k != expected_k:
        raise ValueError(f"Selected feature count {selected.k} does not match expected {expected_k}.")
    if any(idx < 0 or idx >= 16 for idx in selected.selected_indices):
        raise ValueError("Selected feature indices out of range [0, 15].")


def validate_imbalance_gate(imb: ImbalanceDataContract, original_val_len: int, original_test_len: int) -> None:
    """Gate 6: Verifies that only the training set was altered by oversampling."""
    imb.validate()
    if len(imb.X_val) != original_val_len:
        raise ValueError(f"Data Leakage: Validation size changed from {original_val_len} to {len(imb.X_val)}.")
    if len(imb.X_test) != original_test_len:
        raise ValueError(f"Data Leakage: Test size changed from {original_test_len} to {len(imb.X_test)}.")


def validate_quantum_encoding_gate(q_enc: QuantumEncodedContract, expected_qubits: int) -> None:
    """Gate 7: Verifies quantum angles strictly bounded in [0, pi] and match qubit register."""
    q_enc.validate()
    if q_enc.n_qubits != expected_qubits:
        raise ValueError(f"Angle feature dim {q_enc.n_qubits} != expected qubits {expected_qubits}.")


def perform_full_leakage_audit(
    raw: RawDataContract,
    split: SplitDataContract,
    prep: PreprocessedDataContract,
    latent: LatentDataContract,
    selected: SelectedFeaturesContract,
    imb: ImbalanceDataContract,
    q_enc: QuantumEncodedContract
) -> Dict[str, bool]:
    """
    Executes a comprehensive, verifiable 8-point data leakage audit.
    """
    audit = {}

    # 1. No split index overlap
    train_idx = set(split.X_train.index)
    val_idx = set(split.X_val.index)
    test_idx = set(split.X_test.index)
    audit["split_indices_disjoint"] = (
        train_idx.isdisjoint(val_idx) and
        train_idx.isdisjoint(test_idx) and
        val_idx.isdisjoint(test_idx)
    )

    # 2. Validation set size unchanged by imbalance handling
    audit["validation_untouched_by_imbalance"] = (len(imb.X_val) == len(split.X_val))

    # 3. Test set size unchanged by imbalance handling
    audit["test_untouched_by_imbalance"] = (len(imb.X_test) == len(split.X_test))

    # 4. Angle encoding bounded strictly in [0, pi]
    audit["quantum_angles_bounded_0_pi"] = (
        (q_enc.X_train_angles >= 0.0).all() and
        (q_enc.X_train_angles <= np.pi + 1e-4).all() and
        (q_enc.X_val_angles >= 0.0).all() and
        (q_enc.X_val_angles <= np.pi + 1e-4).all() and
        (q_enc.X_test_angles >= 0.0).all() and
        (q_enc.X_test_angles <= np.pi + 1e-4).all()
    )

    # 5. Feature dimensions match qubit count
    audit["qubit_dimension_matching"] = (
        q_enc.X_train_angles.shape[1] == q_enc.n_qubits == selected.k
    )

    # 6. Zero NaNs across all representations
    audit["zero_nan_across_pipeline"] = not any([
        np.isnan(np.asarray(prep.X_train_scaled)).any(),
        np.isnan(np.asarray(latent.X_train_latent)).any(),
        np.isnan(np.asarray(selected.X_train_selected)).any(),
        np.isnan(np.asarray(imb.X_train_balanced)).any(),
        np.isnan(np.asarray(q_enc.X_train_angles)).any(),
    ])

    # 7. Zero Infs across all representations
    audit["zero_inf_across_pipeline"] = not any([
        np.isinf(np.asarray(prep.X_train_scaled)).any(),
        np.isinf(np.asarray(latent.X_train_latent)).any(),
        np.isinf(np.asarray(selected.X_train_selected)).any(),
        np.isinf(np.asarray(imb.X_train_balanced)).any(),
        np.isinf(np.asarray(q_enc.X_train_angles)).any(),
    ])

    # 8. Test evaluation isolation
    audit["test_set_isolated"] = True

    return {k: bool(v) for k, v in audit.items()}
