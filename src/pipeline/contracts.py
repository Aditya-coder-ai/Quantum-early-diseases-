"""
Part 7: Explicit Stage Data Contracts.
Defines strongly-typed, verifiable data contracts passed between pipeline stages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class RawDataContract:
    X: pd.DataFrame
    y: pd.Series
    n_samples: int
    n_features: int
    feature_names: List[str]

    def validate(self) -> None:
        if self.X.empty or self.y.empty:
            raise ValueError("Raw dataset cannot be empty.")
        if len(self.X) != len(self.y):
            raise ValueError(f"Feature length {len(self.X)} != target length {len(self.y)}.")
        if set(self.y.unique()) - {0, 1}:
            raise ValueError(f"Target contains invalid classes: {self.y.unique()}. Expected {{0, 1}}.")


@dataclass
class SplitDataContract:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def validate(self) -> None:
        for split_name, (x, y) in [("train", (self.X_train, self.y_train)),
                                   ("val", (self.X_val, self.y_val)),
                                   ("test", (self.X_test, self.y_test))]:
            if len(x) != len(y):
                raise ValueError(f"{split_name} feature rows ({len(x)}) != target rows ({len(y)}).")
        if self.X_train.shape[1] != self.X_val.shape[1] or self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("Feature dimension mismatch across train, val, and test splits.")


@dataclass
class PreprocessedDataContract:
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    feature_names: List[str]
    scaler_artifact_path: str

    def validate(self) -> None:
        for name, data in [("train", self.X_train_scaled), ("val", self.X_val_scaled), ("test", self.X_test_scaled)]:
            arr = np.asarray(data)
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError(f"Preprocessed {name} contains NaN or infinite values.")
        if self.X_train_scaled.shape[1] != self.X_val_scaled.shape[1]:
            raise ValueError("Preprocessed feature dimensions do not match between train and validation.")


@dataclass
class LatentDataContract:
    X_train_latent: np.ndarray
    X_val_latent: np.ndarray
    X_test_latent: np.ndarray
    latent_dim: int
    model_artifact_path: str

    def validate(self) -> None:
        for name, arr in [("train", self.X_train_latent), ("val", self.X_val_latent), ("test", self.X_test_latent)]:
            if arr.shape[1] != self.latent_dim:
                raise ValueError(f"{name} latent dim {arr.shape[1]} != expected {self.latent_dim}.")
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError(f"{name} latent features contain NaN or infinite values.")


@dataclass
class SelectedFeaturesContract:
    X_train_selected: np.ndarray
    X_val_selected: np.ndarray
    X_test_selected: np.ndarray
    selected_indices: List[int]
    selected_names: List[str]
    method: str
    k: int
    artifact_path: str

    def validate(self) -> None:
        if len(self.selected_indices) != self.k:
            raise ValueError(f"Selected indices count {len(self.selected_indices)} != k {self.k}.")
        for name, arr in [("train", self.X_train_selected), ("val", self.X_val_selected), ("test", self.X_test_selected)]:
            if arr.shape[1] != self.k:
                raise ValueError(f"{name} selected feature dim {arr.shape[1]} != expected {self.k}.")


@dataclass
class ImbalanceDataContract:
    X_train_balanced: np.ndarray
    y_train_balanced: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    method: str
    class_weights: Optional[Dict[int, float]] = None

    def validate(self) -> None:
        if len(self.X_train_balanced) != len(self.y_train_balanced):
            raise ValueError("Balanced train X and y lengths do not match.")
        if self.X_train_balanced.shape[1] != self.X_val.shape[1]:
            raise ValueError("Feature dimension mismatch between balanced train and validation sets.")


@dataclass
class QuantumEncodedContract:
    X_train_angles: np.ndarray
    X_val_angles: np.ndarray
    X_test_angles: np.ndarray
    angle_scaler_path: str
    n_qubits: int

    def validate(self) -> None:
        for name, arr in [("train", self.X_train_angles), ("val", self.X_val_angles), ("test", self.X_test_angles)]:
            if arr.shape[1] != self.n_qubits:
                raise ValueError(f"{name} angle features {arr.shape[1]} != expected qubits {self.n_qubits}.")
            if (arr < 0.0).any() or (arr > np.pi + 1e-4).any():
                raise ValueError(f"{name} angle values exceed strict quantum rotation bounds [0, pi].")


@dataclass
class ModelPredictionContract:
    y_pred: np.ndarray
    y_probs: np.ndarray
    metrics: Dict[str, Any]
    inference_time_s: float
    model_name: str

    def validate(self) -> None:
        if len(self.y_pred) != len(self.y_probs):
            raise ValueError("Prediction and probability vector lengths do not match.")
        if (self.y_probs < 0.0).any() or (self.y_probs > 1.0).any():
            raise ValueError("Predicted probabilities violate [0, 1] range bounds.")


@dataclass
class PipelineReportContract:
    experiment_id: str
    config_hash: str
    status: str
    total_runtime_s: float
    stage_timings: Dict[str, float]
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    resource_profile: Dict[str, Any]
    artifacts: Dict[str, str]
    leakage_audit: Dict[str, bool]
