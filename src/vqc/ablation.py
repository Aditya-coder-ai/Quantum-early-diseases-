"""
Part 6: Systematic Ablation Study for VQC.
Executes controlled experiments across:
    1. Classical vs. VQC baselines
    2. Imbalance strategies (Original, Class Weights, SMOTE, QGAN)
    3. Feature / qubit counts (K in {4, 6, 8})
    4. Circuit depth scaling (L in {1, 2, 3})
    5. Ansatz architectures (Ansatz A vs Ansatz B)
    6. Multi-seed stability (seeds {42, 43, 44})
    7. Noise model robustness (Ideal vs Shots vs Depolarizing)
    8. Final single-pass test set evaluation
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

from src.vqc.config import VQCConfig
from src.vqc.encoding import AngleScaler, FeatureQubitMapper
from src.vqc.circuit import count_quantum_resources
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.training import train_vqc
from src.vqc.evaluation import evaluate_vqc_model, train_and_evaluate_classical_reference
from src.vqc.noise import evaluate_under_noise


def run_vqc_experiment(
    X_train_angles: np.ndarray,
    y_train: np.ndarray,
    X_val_angles: np.ndarray,
    y_val: np.ndarray,
    n_qubits: int,
    n_layers: int = 2,
    ansatz_type: str = "ansatz_b",
    entanglement: str = "ring",
    class_weights: Any = None,
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 0.02,
    early_stopping_patience: int = 8,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[VariationalQuantumClassifier, Dict[str, Any], Dict[str, Any]]:
    """Helper to initialize, train, and evaluate a single VQC model configuration."""
    model = VariationalQuantumClassifier(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz_type=ansatz_type,
        entanglement=entanglement,
        seed=seed,
    )

    train_res = train_vqc(
        model,
        X_train=X_train_angles,
        y_train=y_train,
        X_val=X_val_angles,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        class_weights=class_weights,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
    )

    eval_res = evaluate_vqc_model(model, X_val_angles, y_val)
    eval_res["training_time_s"] = train_res["total_training_time_s"]
    eval_res["epochs_trained"] = train_res["total_epochs_trained"]
    eval_res["best_epoch"] = train_res["best_epoch"]
    eval_res["qubits"] = n_qubits
    eval_res["depth"] = n_layers
    eval_res["ansatz"] = ansatz_type
    eval_res["entanglement"] = entanglement
    eval_res["parameters"] = model.n_params
    eval_res["seed"] = seed

    return model, eval_res, train_res
