"""
Part 6: Variational Quantum Classifier (VQC) Configuration.
Centralizes paths, hyperparameters, architecture configurations, and seeds.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    PROJECT_ROOT, RESULTS_DIR, MODELS_DIR, DATA_PROCESSED_DIR,
    RANDOM_SEED
)

# ── Directories ──────────────────────────────────────────────────────
PART6_RESULTS_DIR = os.path.join(RESULTS_DIR, "part6_vqc")
PART6_MODELS_DIR = os.path.join(MODELS_DIR, "vqc")
PART6_PLOTS_DIR = os.path.join(PART6_RESULTS_DIR, "plots")
PART6_EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "vqc")

for d in [PART6_RESULTS_DIR, PART6_MODELS_DIR, PART6_PLOTS_DIR, PART6_EXPERIMENTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Feature & Data Paths ─────────────────────────────────────────────
QAOA_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "selected", "qaoa")
CLASSICAL_MI_DIR = os.path.join(PROJECT_ROOT, "features", "selected", "classical_mi")
SMOTE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "augmented", "smote")
QGAN_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "augmented", "qgan")


@dataclass
class VQCConfig:
    """Central configuration class for Part 6 VQC experiments."""
    # Data & target specifications
    minority_class: int = 0  # Malignant
    majority_class: int = 1  # Benign
    class_weights: Dict[int, float] = field(default_factory=lambda: {0: 1.3446, 1: 0.7960})

    # Architecture
    default_n_qubits: int = 8
    feature_counts: List[int] = field(default_factory=lambda: [4, 6, 8])
    default_circuit_depth: int = 2
    circuit_depths: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Ansatz and Entanglement
    # ansatz_a: RY rotations + linear CNOT entanglement
    # ansatz_b: RY, RZ rotations + ring/circular CNOT entanglement
    ansatz_type: str = "ansatz_b"
    entanglement: str = "ring"
    measurement_type: str = "expval_z0"  # or 'average_z'

    # Simulation engine
    preferred_device: str = "lightning.qubit"
    fallback_device: str = "default.qubit"
    diff_method: str = "adjoint"

    # Noise model settings
    noisy_shots: int = 1024
    depolarizing_prob: float = 0.01

    # Training settings
    learning_rate: float = 0.02
    batch_size: int = 16
    epochs: int = 35
    early_stopping_patience: int = 8
    weight_decay: float = 1e-4

    # Reproducibility
    seed: int = RANDOM_SEED
    stability_seeds: List[int] = field(default_factory=lambda: [42, 43, 44])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config dataclass to dictionary."""
        return {
            "minority_class": self.minority_class,
            "majority_class": self.majority_class,
            "class_weights": self.class_weights,
            "default_n_qubits": self.default_n_qubits,
            "feature_counts": self.feature_counts,
            "default_circuit_depth": self.default_circuit_depth,
            "circuit_depths": self.circuit_depths,
            "ansatz_type": self.ansatz_type,
            "entanglement": self.entanglement,
            "measurement_type": self.measurement_type,
            "preferred_device": self.preferred_device,
            "fallback_device": self.fallback_device,
            "diff_method": self.diff_method,
            "noisy_shots": self.noisy_shots,
            "depolarizing_prob": self.depolarizing_prob,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
            "stability_seeds": self.stability_seeds,
        }
