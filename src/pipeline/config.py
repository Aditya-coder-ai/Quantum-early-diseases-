"""
Part 7: Pipeline Configuration, Schema Validation & Fingerprinting.
Provides typed configuration management and SHA-256 fingerprinting for experiment tracking.
"""
from __future__ import annotations

import os
import json
import hashlib
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import PROJECT_ROOT, RANDOM_SEED


@dataclass
class PipelineConfig:
    experiment_name: str = "hybrid_qaoa_smote_vqc"
    seed: int = RANDOM_SEED

    # Dataset & Splitting
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Preprocessing
    scaling_method: str = "standard"  # 'standard', 'robust', 'minmax'

    # Feature Extraction (Part 3 Autoencoder)
    latent_dim: int = 16
    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    ae_epochs: int = 40
    ae_batch_size: int = 32
    ae_lr: float = 0.001

    # Feature Selection (Part 4 QAOA vs Classical)
    feature_selection_method: str = "qaoa"  # 'qaoa', 'mutual_info', 'rfe', 'none'
    num_selected_features: int = 8

    # Class Imbalance Handling (Part 5)
    imbalance_method: str = "smote"  # 'none', 'class_weight', 'smote', 'qgan'
    imbalance_ratio: float = 1.0     # 1.0 brings minority to 50/50 parity

    # Quantum & VQC (Part 6)
    preferred_device: str = "lightning.qubit"
    fallback_device: str = "default.qubit"
    diff_method: str = "adjoint"
    vqc_layers: int = 2
    vqc_ansatz: str = "ansatz_b"
    vqc_entanglement: str = "ring"
    vqc_lr: float = 0.02
    vqc_batch_size: int = 16
    vqc_epochs: int = 35
    vqc_patience: int = 8
    vqc_shots: Optional[int] = None

    # Classical Reference Model
    classical_baseline: str = "SVM_RBF"

    # Artifact Storage
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "artifacts", "experiments")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to clean dictionary."""
        return asdict(self)

    def get_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint hash for experiment tracking."""
        d = self.to_dict()
        # Exclude artifacts_dir from hash to remain filesystem agnostic
        d.pop("artifacts_dir", None)
        serialized = json.dumps(d, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def save_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
