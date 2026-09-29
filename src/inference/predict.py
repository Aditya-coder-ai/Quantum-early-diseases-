"""
Part 7: Production-Ready Standalone Inference Pipeline.
Loads serialized pipeline artifacts and classifies new, unlabelled clinical samples.
Guarantees zero leakage: never applies SMOTE or QGAN during inference.
"""
from __future__ import annotations

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Union, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.config import LATENT_DIM, AE_HIDDEN_DIMS
from src.preprocessing.pipeline import MedicalPreprocessingPipeline
from src.models.autoencoder import Autoencoder
from src.vqc.encoding import AngleScaler
from src.vqc.model import VariationalQuantumClassifier


class MedicalInferenceEngine:
    """
    Lightweight, dependency-isolated inference engine that applies the fitted
    hybrid classical-quantum pipeline to raw patient clinical feature records.
    """
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load and verify all required pipeline artifacts from artifact_dir."""
        # 1. Preprocessing pipeline
        prep_path = os.path.join(self.artifact_dir, "preprocessing_pipeline.joblib")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Missing preprocessing artifact: {prep_path}")
        self.preprocessor = MedicalPreprocessingPipeline.load(prep_path)

        # 2. Autoencoder feature extractor
        ae_path = os.path.join(self.artifact_dir, "autoencoder.pth")
        if not os.path.exists(ae_path):
            raise FileNotFoundError(f"Missing autoencoder artifact: {ae_path}")
        
        input_dim = len(self.preprocessor.feature_names_in_)
        self.autoencoder = Autoencoder(input_dim=input_dim, hidden_dims=AE_HIDDEN_DIMS, latent_dim=LATENT_DIM)
        ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.eval()

        # 3. Feature selection metadata
        fs_path = os.path.join(self.artifact_dir, "feature_selection.json")
        if not os.path.exists(fs_path):
            raise FileNotFoundError(f"Missing feature selection metadata: {fs_path}")
        with open(fs_path, "r") as f:
            fs_data = json.load(f)
        self.selected_indices = fs_data["selected_indices"]
        self.selected_names = fs_data["selected_names"]
        self.k = len(self.selected_indices)

        # 4. Quantum angle scaler
        scaler_path = os.path.join(self.artifact_dir, "angle_scaler.json")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing angle scaler artifact: {scaler_path}")
        self.angle_scaler = AngleScaler.load(scaler_path)

        # 5. Trained VQC model checkpoint
        vqc_path = os.path.join(self.artifact_dir, "vqc_model.pt")
        if not os.path.exists(vqc_path):
            raise FileNotFoundError(f"Missing VQC checkpoint: {vqc_path}")
        self.vqc_model = VariationalQuantumClassifier.load(vqc_path, map_location="cpu")
        self.vqc_model.eval()

        self.is_loaded = True

    def predict(
        self,
        X_raw: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        End-to-end inference pass from raw 30-dimensional features to class prediction.
        
        Args:
            X_raw: Input raw patient clinical features (DataFrame, ndarray, or dict)
            threshold: Probability threshold for classification (default 0.5)
        
        Returns:
            Structured diagnostic assessment dictionary.
        """
        t0 = time.time()

        if isinstance(X_raw, dict):
            X_df = pd.DataFrame([X_raw])
        elif isinstance(X_raw, np.ndarray):
            X_df = pd.DataFrame(X_raw, columns=self.preprocessor.feature_names_in_)
        elif isinstance(X_raw, pd.DataFrame):
            X_df = X_raw.copy()
        else:
            raise TypeError("Input must be a DataFrame, numpy array, or dictionary.")

        n_samples = len(X_df)

        # 1. Classical Preprocessing (zero-leakage fitted scaler)
        X_scaled = np.asarray(self.preprocessor.transform(X_df))

        # 2. Classical Representation (Autoencoder 30D -> 16D)
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            z_latent = self.autoencoder.encode(x_tensor).numpy()

        # 3. Compact Feature Selection (16D -> 8D)
        z_selected = z_latent[:, self.selected_indices]

        # 4. Quantum Normalization & Angle Encoding ([0, pi])
        angles = self.angle_scaler.transform(z_selected)

        # 5. VQC Quantum Circuit Inference
        benign_probs = self.vqc_model.predict_proba(angles)
        malignant_probs = 1.0 - benign_probs

        # 0 = Malignant, 1 = Benign
        predicted_classes = (benign_probs >= threshold).astype(int)
        class_labels = ["Malignant" if c == 0 else "Benign" for c in predicted_classes]
        risk_levels = ["HIGH_RISK_MALIGNANT" if p >= threshold else "LOW_RISK_BENIGN" for p in malignant_probs]

        latency_total = time.time() - t0
        latency_per_sample_ms = (latency_total / n_samples) * 1000

        return {
            "n_samples": n_samples,
            "predicted_classes": predicted_classes.tolist(),
            "diagnosis_labels": class_labels,
            "malignant_probabilities": np.round(malignant_probs, 4).tolist(),
            "benign_probabilities": np.round(benign_probs, 4).tolist(),
            "oncology_risk_assessment": risk_levels,
            "features_used": self.selected_names,
            "qubit_count": self.k,
            "total_inference_time_s": round(latency_total, 4),
            "latency_per_sample_ms": round(latency_per_sample_ms, 2),
        }
