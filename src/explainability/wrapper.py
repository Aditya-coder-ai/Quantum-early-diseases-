"""
Part 8: Clean Prediction Wrappers for Model Explainability.
Decouples SHAP and attribution algorithms from quantum circuit internals,
exposing standard (N, 2) probability interfaces: [P(Malignant), P(Benign)].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Any, Optional, Union, List

from src.vqc.model import VariationalQuantumClassifier
from src.vqc.encoding import AngleScaler


class VQCPredictionWrapper:
    """
    Standardized probability wrapper for the Variational Quantum Classifier.
    Transforms numerical selected features into quantum angles and computes
    disease malignancy probabilities.
    
    Output convention:
        Column 0: P(class=0) = P(Malignant)  [Disease Positive]
        Column 1: P(class=1) = P(Benign)     [Disease Negative]
    """
    def __init__(
        self,
        vqc_model: VariationalQuantumClassifier,
        angle_scaler: Optional[AngleScaler] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.vqc_model = vqc_model
        self.angle_scaler = angle_scaler
        self.feature_names = feature_names or [f"selected_{i}" for i in range(vqc_model.n_qubits)]
        self.vqc_model.eval()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes 2-class probabilities [P(Malignant), P(Benign)].
        
        Args:
            X: Input selected features of shape (N, n_qubits) or DataFrame.
        Returns:
            np.ndarray of shape (N, 2)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Scale features to [0, pi] angles if scaler is provided
        if self.angle_scaler is not None:
            angles = self.angle_scaler.transform(X_arr)
        else:
            angles = X_arr

        # VQC predict_proba outputs P(class=1, Benign)
        p_benign = self.vqc_model.predict_proba(angles)
        p_malignant = 1.0 - p_benign

        return np.column_stack([p_malignant, p_benign])

    def predict_malignant_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convenience function returning 1D array of malignancy risk probabilities for SHAP."""
        return self.predict_proba(X)[:, 0]

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """Binary diagnosis predictions: 0=Malignant, 1=Benign."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)


class ClassicalPredictionWrapper:
    """
    Standardized probability wrapper for classical reference classifiers (e.g. SVM, RF).
    """
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        minority_class: int = 0
    ):
        self.model = model
        self.feature_names = feature_names
        self.minority_class = minority_class

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_arr)
            # If classes_ is [0, 1], column 0 is class 0 (Malignant)
            if hasattr(self.model, "classes_") and self.model.classes_[0] != self.minority_class:
                probs = np.column_stack([probs[:, 1], probs[:, 0]])
            return probs
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_arr)
            # Sigmoid mapping for SVM decision boundary
            p_class1 = 1.0 / (1.0 + np.exp(-scores))
            p_class0 = 1.0 - p_class1
            return np.column_stack([p_class0, p_class1])
        else:
            preds = self.model.predict(X_arr)
            return np.column_stack([1.0 - preds, preds])

    def predict_malignant_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.predict_proba(X)[:, 0]

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.predict_proba(X)[:, 1] >= 0.5


class EndToEndHybridWrapper:
    """
    Full pipeline wrapper taking raw clinical features (30D) and returning
    quantum predictions, enabling raw-space SHAP analysis.
    """
    def __init__(
        self,
        inference_engine: Any,
    ):
        self.inference_engine = inference_engine
        self.feature_names = list(inference_engine.preprocessor.feature_names_in_)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X.copy()

        res = self.inference_engine.predict(X_df)
        p_mal = np.array(res["malignant_probabilities"])
        p_ben = np.array(res["benign_probabilities"])
        return np.column_stack([p_mal, p_ben])

    def predict_malignant_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.predict_proba(X)[:, 0]
