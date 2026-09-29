"""
Configuration schema and parameters for Part 9: Classical vs Hybrid Comparison.
"""
from __future__ import annotations

import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


def _clean_for_yaml(obj: Any) -> Any:
    """Recursively converts tuples to lists and clean primitive types for safe YAML serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_yaml(v) for v in obj]
    return obj


@dataclass
class ModelConfig:
    model_id: str
    model_name: str
    model_family: str  # 'classical', 'deep_learning', 'compact_classical', 'hybrid_vqc'
    input_space: str   # 'raw_30', 'latent_16', 'selected_8_classical', 'selected_8_qaoa'
    feature_selection_method: str = "none"  # 'none', 'mutual_info', 'qaoa'
    imbalance_method: str = "none"          # 'none', 'class_weight', 'smote', 'qgan'
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    n_features: int = 30
    n_qubits: int = 0
    circuit_depth: int = 0


@dataclass
class ComparisonConfig:
    experiment_name: str = "classical_vs_hybrid_comparison"
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    primary_seed: int = 42
    
    # Dataset & Feature paths
    dataset_dir: str = "data/processed"
    features_dir: str = "features/selected"
    artifacts_dir: str = "experiments/comparison"
    
    # Threshold Tuning Policy
    threshold_tuning: bool = True
    threshold_metric: str = "f1"  # 'f1', 'balanced_accuracy', 'youden_j'
    min_threshold: float = 0.1
    max_threshold: float = 0.9
    threshold_steps: int = 41
    
    # VQC Hyperparameters
    vqc_epochs: int = 20
    vqc_patience: int = 6
    vqc_lr: float = 0.02
    vqc_batch_size: int = 16
    vqc_layers: int = 2
    vqc_ansatz: str = "ansatz_b"
    vqc_entanglement: str = "ring"
    preferred_device: str = "lightning.qubit"
    fallback_device: str = "default.qubit"
    diff_method: str = "adjoint"
    vqc_shots: Optional[int] = None
    
    # Ablation Settings
    ablation_feature_counts: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12])
    ablation_circuit_depths: List[int] = field(default_factory=lambda: [1, 2, 3])
    ablation_data_fractions: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 1.0])
    robustness_noise_sigmas: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.2])
    quantum_noise_shots: List[int] = field(default_factory=lambda: [1024, 4096])
    quantum_depolarizing_probs: List[float] = field(default_factory=lambda: [0.01, 0.02])
    
    # Execution mode
    quick_mode: bool = False
    
    # Model catalog
    models: List[ModelConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.models:
            self.models = self._default_models()

    def _default_models(self) -> List[ModelConfig]:
        return [
            ModelConfig(
                model_id="logreg_raw",
                model_name="Logistic Regression",
                model_family="classical",
                input_space="raw_30",
                n_features=30,
                hyperparameters={"C": 1.0, "max_iter": 1000}
            ),
            ModelConfig(
                model_id="svm_rbf_raw",
                model_name="SVM (RBF)",
                model_family="classical",
                input_space="raw_30",
                n_features=30,
                hyperparameters={"C": 1.0, "kernel": "rbf", "probability": True}
            ),
            ModelConfig(
                model_id="rf_raw",
                model_name="Random Forest",
                model_family="classical",
                input_space="raw_30",
                n_features=30,
                hyperparameters={"n_estimators": 100, "max_depth": 5}
            ),
            ModelConfig(
                model_id="mlp_raw",
                model_name="Classical MLP",
                model_family="deep_learning",
                input_space="raw_30",
                n_features=30,
                hyperparameters={"hidden_layer_sizes": [32, 16], "max_iter": 200}
            ),
            ModelConfig(
                model_id="compact_svm_classical_fs",
                model_name="Compact Classical (MI-SVM)",
                model_family="compact_classical",
                input_space="selected_8_classical",
                feature_selection_method="mutual_info",
                n_features=8,
                hyperparameters={"C": 1.0, "kernel": "rbf", "probability": True}
            ),
            ModelConfig(
                model_id="compact_svm_qaoa_fs",
                model_name="Compact Classical (QAOA-SVM)",
                model_family="compact_classical",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                n_features=8,
                hyperparameters={"C": 1.0, "kernel": "rbf", "probability": True}
            ),
            ModelConfig(
                model_id="hybrid_vqc_original",
                model_name="Hybrid VQC (Unweighted)",
                model_family="hybrid_vqc",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                imbalance_method="none",
                n_features=8,
                n_qubits=8,
                circuit_depth=2
            ),
            ModelConfig(
                model_id="hybrid_vqc_smote",
                model_name="Hybrid VQC + SMOTE",
                model_family="hybrid_vqc",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                imbalance_method="smote",
                n_features=8,
                n_qubits=8,
                circuit_depth=2
            ),
            ModelConfig(
                model_id="hybrid_vqc_qgan",
                model_name="Hybrid VQC + QGAN",
                model_family="hybrid_vqc",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                imbalance_method="qgan",
                n_features=8,
                n_qubits=8,
                circuit_depth=2
            ),
        ]

    @classmethod
    def from_yaml(cls, path: str) -> "ComparisonConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        models_raw = raw.pop("models", None)
        cfg = cls(**raw)
        if models_raw:
            cfg.models = [ModelConfig(**m) for m in models_raw]
        return cfg

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = _clean_for_yaml(asdict(self))
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
