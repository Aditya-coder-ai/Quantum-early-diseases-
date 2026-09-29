"""
app/services/inference_service.py
Core inference service handling feature translation, tensor encoding,
and hybrid variational quantum classification.
Guarantees zero training, zero SMOTE, and zero QGAN execution during inference requests.
"""
from __future__ import annotations

import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union, Optional

from app.services.model_manager import model_manager
from app.schemas.prediction import (
    PredictionRequest, SinglePredictionResult,
    BatchPredictionRequest, BatchPredictionResponse
)
from app.core.errors import ModelNotReadyError, InvalidInputError, BatchSizeExceededError
from app.config import settings
from src.security.validation import WDBC_FEATURE_NAMES, MAX_BATCH_SIZE


class InferenceService:
    """Orchestrates validation, preprocessing, latent extraction, and quantum evaluation."""
    
    def __init__(self, manager=model_manager):
        self.manager = manager

    def _ensure_dataframe(
        self,
        features: Union[List[float], Dict[str, float], List[Union[List[float], Dict[str, float]]]]
    ) -> pd.DataFrame:
        """Converts raw list or dict feature inputs into a canonical 30D Pandas DataFrame."""
        if not self.manager.is_loaded:
            raise ModelNotReadyError()

        feature_names = list(self.manager.preprocessor.feature_names_in_)

        # Single sample as dictionary
        if isinstance(features, dict):
            return pd.DataFrame([features])[feature_names]
        
        # Single sample as list of 30 floats
        elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], (int, float)):
            if len(features) != len(feature_names):
                raise InvalidInputError(f"Expected {len(feature_names)} features, got {len(features)}")
            return pd.DataFrame([features], columns=feature_names)

        # Batch of samples (list of lists or list of dicts)
        elif isinstance(features, list):
            rows = []
            for idx, item in enumerate(features):
                if isinstance(item, dict):
                    rows.append(item)
                elif isinstance(item, list):
                    if len(item) != len(feature_names):
                        raise InvalidInputError(f"Batch sample {idx} has {len(item)} features, expected {len(feature_names)}")
                    rows.append(dict(zip(feature_names, item)))
                else:
                    raise InvalidInputError(f"Invalid format for batch sample {idx}")
            return pd.DataFrame(rows)[feature_names]

        raise InvalidInputError("Unsupported feature data structure")

    def predict_single(
        self,
        request: PredictionRequest,
        request_id: str
    ) -> Tuple[SinglePredictionResult, Dict[str, Any]]:
        """Executes full diagnostic pipeline for an individual patient record."""
        t0 = time.perf_counter()
        
        # 1. Parse and validate input
        df_raw = self._ensure_dataframe(request.features)
        
        # 2. Select decision threshold
        threshold = request.threshold if request.threshold is not None else self.manager.calibrated_threshold

        with self.manager.inference_lock:
            # 3. Classical Preprocessing (zero-leakage fitted scaler)
            X_scaled = np.asarray(self.manager.preprocessor.transform(df_raw))

            # 4. Classical Feature Extraction (Autoencoder 30D -> 16D)
            with torch.no_grad():
                x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                z_latent = self.manager.autoencoder.encode(x_tensor).numpy()

            # 5. Compact Feature Selection (16D -> 8D)
            z_selected = z_latent[:, self.manager.selected_indices]

            # 6. Quantum Normalization ([0, pi])
            angles = self.manager.angle_scaler.transform(z_selected)

            # 7. Variational Quantum Circuit Evaluation
            benign_prob = float(self.manager.vqc_model.predict_proba(angles)[0])
            malignant_prob = float(np.clip(1.0 - benign_prob, 0.0, 1.0))

        # Binary classification rule: benign_prob >= threshold implies Benign (1), else Malignant (0)
        predicted_class = 1 if benign_prob >= threshold else 0
        diagnosis_label = "Benign" if predicted_class == 1 else "Malignant"
        risk_assessment = "HIGH_RISK_MALIGNANT" if malignant_prob >= (1.0 - threshold) else "LOW_RISK_BENIGN"

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        result = SinglePredictionResult(
            predicted_class=predicted_class,
            diagnosis_label=diagnosis_label,
            malignant_probability=round(malignant_prob, 4),
            benign_probability=round(benign_prob, 4),
            oncology_risk_assessment=risk_assessment,
            decision_threshold_used=round(threshold, 4)
        )

        metadata = {
            "inference_time_ms": round(total_latency_ms, 2),
            "quantum_qubits": self.manager.vqc_model.n_qubits,
            "circuit_depth": self.manager.vqc_model.n_layers,
            "features_evaluated": self.manager.selected_names,
            "backend": settings.quantum_backend,
            "sample_id": request.sample_id
        }

        return result, metadata

    def predict_batch(
        self,
        request: BatchPredictionRequest,
        request_id: str
    ) -> BatchPredictionResponse:
        """Executes vectorized batch inference for multiple patient records."""
        n_samples = len(request.samples)
        if n_samples > settings.max_batch_size:
            raise BatchSizeExceededError(size=n_samples, max_size=settings.max_batch_size)

        t0 = time.perf_counter()
        
        # 1. Parse into single batch DataFrame
        df_raw = self._ensure_dataframe(request.samples)
        threshold = request.threshold if request.threshold is not None else self.manager.calibrated_threshold

        with self.manager.inference_lock:
            # 2. Batch preprocessing
            X_scaled = np.asarray(self.manager.preprocessor.transform(df_raw))

            # 3. Batch autoencoder extraction
            with torch.no_grad():
                x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                z_latent = self.manager.autoencoder.encode(x_tensor).numpy()

            # 4. Batch feature selection
            z_selected = z_latent[:, self.manager.selected_indices]

            # 5. Batch angle scaling
            angles = self.manager.angle_scaler.transform(z_selected)

            # 6. Batch VQC evaluation
            benign_probs = self.manager.vqc_model.predict_proba(angles)
            malignant_probs = np.clip(1.0 - benign_probs, 0.0, 1.0)

        # 7. Collect results
        predictions: List[SinglePredictionResult] = []
        for b_prob, m_prob in zip(benign_probs, malignant_probs):
            pred_class = 1 if b_prob >= threshold else 0
            label = "Benign" if pred_class == 1 else "Malignant"
            risk = "HIGH_RISK_MALIGNANT" if m_prob >= (1.0 - threshold) else "LOW_RISK_BENIGN"
            predictions.append(
                SinglePredictionResult(
                    predicted_class=pred_class,
                    diagnosis_label=label,
                    malignant_probability=round(float(m_prob), 4),
                    benign_probability=round(float(b_prob), 4),
                    oncology_risk_assessment=risk,
                    decision_threshold_used=round(threshold, 4)
                )
            )

        total_time_s = time.perf_counter() - t0
        latency_per_sample_ms = (total_time_s / n_samples) * 1000.0

        return BatchPredictionResponse(
            success=True,
            request_id=request_id,
            count=n_samples,
            predictions=predictions,
            total_inference_time_s=round(total_time_s, 4),
            latency_per_sample_ms=round(latency_per_sample_ms, 2),
            model={"id": self.manager.model_id, "version": self.manager.model_version}
        )


inference_service = InferenceService()
