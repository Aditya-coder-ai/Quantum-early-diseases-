"""
app/services/model_manager.py
Singleton lifecycle manager for hybrid classical-quantum model artifacts.
Loads all serialized components ONCE at application startup and enforces thread safety.
"""
from __future__ import annotations

import os
import json
import threading
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from configs.config import PROJECT_ROOT, LATENT_DIM, AE_HIDDEN_DIMS
from src.preprocessing.pipeline import MedicalPreprocessingPipeline
from src.models.autoencoder import Autoencoder
from src.vqc.encoding import AngleScaler
from src.vqc.model import VariationalQuantumClassifier
from src.explainability.wrapper import VQCPredictionWrapper, EndToEndHybridWrapper
from src.explainability.shap_explainer import HybridSHAPExplainer
from app.core.logging import api_logger
from app.core.errors import ModelNotReadyError, ArtifactNotFoundError


class ModelManager:
    """
    Central manager responsible for loading, validating, and holding in memory
    the complete hybrid classical-quantum medical diagnostic pipeline.
    """
    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, artifact_dir: Optional[str] = None):
        if self._initialized:
            return

        self.artifact_dir = artifact_dir or os.path.join(
            PROJECT_ROOT, "artifacts", "experiments", "config_b_qaoa_vqc_orig_fa7a21e45376"
        )
        self.is_loaded = False
        self.artifacts_valid = False
        self.load_error: Optional[str] = None
        self.inference_lock = threading.Lock()
        
        # Pipeline components
        self.preprocessor: Optional[MedicalPreprocessingPipeline] = None
        self.autoencoder: Optional[Autoencoder] = None
        self.selected_indices: List[int] = []
        self.selected_names: List[str] = []
        self.angle_scaler: Optional[AngleScaler] = None
        self.vqc_model: Optional[VariationalQuantumClassifier] = None
        self.calibrated_threshold: float = 0.5
        self.model_id: str = "hybrid_vqc_primary"
        self.model_version: str = "1.0.0"
        
        # Explainability engine
        self.shap_explainer: Optional[HybridSHAPExplainer] = None
        
        self._initialized = True

    def load_artifacts(self, artifact_dir: Optional[str] = None) -> bool:
        """
        Loads and validates all pipeline artifacts from disk.
        Returns True if successful, raises exception or records load_error if failed.
        """
        if artifact_dir:
            self.artifact_dir = artifact_dir

        api_logger.info(f"Loading hybrid quantum model artifacts from: {self.artifact_dir}")
        t0 = time.time()

        try:
            # 1. Preprocessing Pipeline
            prep_path = os.path.join(self.artifact_dir, "preprocessing_pipeline.joblib")
            if not os.path.exists(prep_path):
                # Fallback to models root
                prep_path = os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.joblib")
            if not os.path.exists(prep_path):
                raise ArtifactNotFoundError(f"preprocessing_pipeline.joblib at {prep_path}")
            self.preprocessor = MedicalPreprocessingPipeline.load(prep_path)

            # 2. Autoencoder Feature Extractor (30 -> 16)
            ae_path = os.path.join(self.artifact_dir, "autoencoder.pth")
            if not os.path.exists(ae_path):
                ae_path = os.path.join(PROJECT_ROOT, "models", "autoencoder.pth")
            if not os.path.exists(ae_path):
                raise ArtifactNotFoundError(f"autoencoder.pth at {ae_path}")

            input_dim = len(self.preprocessor.feature_names_in_)
            self.autoencoder = Autoencoder(input_dim=input_dim, hidden_dims=AE_HIDDEN_DIMS, latent_dim=LATENT_DIM)
            ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.autoencoder.load_state_dict(state_dict)
            self.autoencoder.eval()

            # 3. Feature Selection Metadata
            fs_path = os.path.join(self.artifact_dir, "feature_selection.json")
            if not os.path.exists(fs_path):
                # Default canonical QAOA indices if file missing in experiment folder
                self.selected_indices = [0, 1, 3, 4, 6, 8, 9, 14]
                self.selected_names = [f"latent_{i}" for i in self.selected_indices]
            else:
                with open(fs_path, "r") as f:
                    fs_data = json.load(f)
                self.selected_indices = fs_data.get("selected_indices", [0, 1, 3, 4, 6, 8, 9, 14])
                self.selected_names = fs_data.get("selected_names", [f"latent_{i}" for i in self.selected_indices])

            # 4. Quantum Angle Scaler
            scaler_path = os.path.join(self.artifact_dir, "angle_scaler.json")
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(PROJECT_ROOT, "models", "angle_scaler.json")
            if not os.path.exists(scaler_path):
                raise ArtifactNotFoundError(f"angle_scaler.json at {scaler_path}")
            self.angle_scaler = AngleScaler.load(scaler_path)

            # 5. Trained VQC Model Checkpoint
            vqc_path = os.path.join(self.artifact_dir, "vqc_model.pt")
            if not os.path.exists(vqc_path):
                vqc_path = os.path.join(PROJECT_ROOT, "models", "vqc_model.pt")
            if not os.path.exists(vqc_path):
                raise ArtifactNotFoundError(f"vqc_model.pt at {vqc_path}")
            self.vqc_model = VariationalQuantumClassifier.load(vqc_path, map_location="cpu")
            self.vqc_model.eval()

            # 6. Verify Dimensional Compatibility
            assert len(self.preprocessor.feature_names_in_) == 30, "Preprocessor must accept 30 features"
            assert self.autoencoder.input_dim == 30, "Autoencoder must accept 30 inputs"
            assert self.autoencoder.latent_dim == 16, "Autoencoder must output 16 latent dimensions"
            assert len(self.selected_indices) == 8, "Feature selector must select exactly 8 features"
            assert self.vqc_model.n_qubits == 8, "VQC must configure 8 qubits"

            # 7. Load Calibrated Decision Threshold
            results_path = os.path.join(self.artifact_dir, "final_results.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path, "r") as f:
                        res = json.load(f)
                    self.calibrated_threshold = float(res.get("validation_metrics", {}).get("threshold", 0.5))
                    self.model_id = res.get("experiment_id", os.path.basename(self.artifact_dir))
                except Exception:
                    self.calibrated_threshold = 0.5
            else:
                self.calibrated_threshold = 0.5
                self.model_id = os.path.basename(self.artifact_dir)

            # 8. Initialize Reference Background for Fast SHAP Explainability
            self._init_shap_explainer()

            self.is_loaded = True
            self.artifacts_valid = True
            self.load_error = None
            api_logger.info(f"ModelManager successfully initialized in {time.time() - t0:.2f}s")
            return True

        except Exception as e:
            self.is_loaded = False
            self.artifacts_valid = False
            self.load_error = str(e)
            api_logger.error(f"Failed to load model artifacts: {e}")
            raise e

    def _init_shap_explainer(self) -> None:
        """Pre-constructs the SHAP KernelExplainer using a small training-split background."""
        try:
            train_csv = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
            if os.path.exists(train_csv):
                df_train = pd.read_csv(train_csv)
                feature_cols = [c for c in df_train.columns if c not in ("target", "diagnosis", "id", "patient_id")]
                X_train_raw = df_train[feature_cols].values
            else:
                # Synthetic fallback reference within standard range
                X_train_raw = np.zeros((30, 30))

            # Sample 20 representative background points
            rng = np.random.RandomState(42)
            n_bg = min(20, len(X_train_raw))
            bg_indices = rng.choice(len(X_train_raw), size=n_bg, replace=False)
            bg_raw = X_train_raw[bg_indices]

            # Wrapper function for SHAP: Raw 30D -> Malignancy probability
            def predict_malignant_fn(X_batch: np.ndarray) -> np.ndarray:
                return self.predict_malignant_probabilities(X_batch)

            self.shap_explainer = HybridSHAPExplainer(
                predict_fn=predict_malignant_fn,
                X_background=bg_raw,
                feature_names=list(self.preprocessor.feature_names_in_),
                background_size=20,
                model_name="Hybrid_VQC_API",
                seed=42
            )
            api_logger.info("Hybrid SHAP KernelExplainer successfully pre-initialized")
        except Exception as e:
            api_logger.warning(f"Could not pre-initialize SHAP explainer: {e}")
            self.shap_explainer = None

    def predict_malignant_probabilities(self, X_raw: np.ndarray) -> np.ndarray:
        """Fast vectorized helper computing malignancy risk probabilities for SHAP."""
        with self.inference_lock:
            X_scaled = np.asarray(self.preprocessor.transform(X_raw))
            with torch.no_grad():
                x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                z_latent = self.autoencoder.encode(x_tensor).numpy()
            z_selected = z_latent[:, self.selected_indices]
            angles = self.angle_scaler.transform(z_selected)
            benign_probs = self.vqc_model.predict_proba(angles)
            return 1.0 - benign_probs

    def get_metadata(self) -> Dict[str, Any]:
        """Returns safe, unprivileged model architecture and versioning information."""
        if not self.is_loaded:
            raise ModelNotReadyError("Model artifacts not loaded")
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_type": "hybrid_classical_quantum_vqc",
            "input_feature_count": 30,
            "selected_feature_count": len(self.selected_indices),
            "selected_feature_names": self.selected_names,
            "qubits": self.vqc_model.n_qubits,
            "circuit_depth": self.vqc_model.n_layers,
            "trainable_parameters": sum(p.numel() for p in self.vqc_model.parameters() if p.requires_grad),
            "quantum_backend": "PennyLane default.qubit (Simulation)",
            "classification_threshold": self.calibrated_threshold,
            "preprocessing_strategy": "Zero-leakage StandardScaler (fitted strictly on training split)",
            "imbalance_strategy": "Class-isolated synthetic balance on training split",
            "validation_status": "VALIDATED"
        }


model_manager = ModelManager()
