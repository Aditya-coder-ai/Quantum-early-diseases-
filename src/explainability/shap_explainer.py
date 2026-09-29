"""
Part 8: Model-Agnostic SHAP Explainer for Hybrid Classical-Quantum Systems.
Uses shap.KernelExplainer to treat quantum circuits and non-linear pipelines
as pure mathematical probability functions, strictly utilizing training-split backgrounds.
"""
from __future__ import annotations

import time
import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from src.explainability.schemas import LocalExplanation, GlobalFeatureImportance

logger = logging.getLogger(__name__)


class HybridSHAPExplainer:
    """
    SHAP-based attribution engine tailored for hybrid classical-quantum models.
    Operates strictly on training-set background summaries to enforce zero data leakage.
    """
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_background: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        background_size: int = 40,
        model_name: str = "Hybrid_VQC",
        seed: int = 42,
    ):
        """
        Args:
            predict_fn: Function mapping (N, D) -> (N,) malignancy probabilities P(Malignant).
            X_background: Reference data drawn EXCLUSIVELY from training split.
            feature_names: Descriptive names for the D features.
            background_size: Number of background samples to represent baseline expectation.
            model_name: Identifier for model being explained.
            seed: Reproducibility seed for background sampling.
        """
        self.predict_fn = predict_fn
        self.model_name = model_name
        self.seed = seed

        # Convert background to 2D numpy array
        bg_arr = np.asarray(X_background, dtype=np.float64)
        if bg_arr.ndim == 1:
            bg_arr = bg_arr.reshape(1, -1)

        self.n_features = bg_arr.shape[1]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features)]

        # Sample representative background subset (strictly from train split)
        rng = np.random.RandomState(seed)
        if len(bg_arr) > background_size:
            idx = rng.choice(len(bg_arr), size=background_size, replace=False)
            self.background = bg_arr[idx]
        else:
            self.background = bg_arr.copy()

        # Initialize model-agnostic KernelExplainer
        logger.info(f"[SHAP] Initializing KernelExplainer with {len(self.background)} training background points.")
        self.explainer = shap.KernelExplainer(
            self.predict_fn,
            self.background,
            seed=seed
        )

        # Base value represents expected malignancy probability across background
        try:
            self.base_value = float(np.mean(self.predict_fn(self.background)))
        except Exception:
            self.base_value = 0.5

    def explain_instance(
        self,
        x_sample: Union[np.ndarray, pd.Series, List[float]],
        sample_id: Union[int, str] = 0,
        nsamples: int = 100,
    ) -> LocalExplanation:
        """
        Generates local SHAP explanation for an individual patient profile.
        
        Args:
            x_sample: 1D feature array of length n_features.
            sample_id: Identifier for patient/sample.
            nsamples: Number of times to re-evaluate background in KernelExplainer.
            
        Returns:
            LocalExplanation containing positive/negative drivers toward malignancy.
        """
        x_arr = np.asarray(x_sample, dtype=np.float64).ravel()
        if len(x_arr) != self.n_features:
            raise ValueError(f"Sample feature count ({len(x_arr)}) does not match expected ({self.n_features}).")

        # 1. Compute actual model probability
        p_malignant = float(self.predict_fn(x_arr.reshape(1, -1))[0])
        p_benign = 1.0 - p_malignant
        pred_class = 0 if p_malignant >= 0.5 else 1
        class_label = "Malignant" if pred_class == 0 else "Benign"

        # 2. Compute Kernel SHAP values
        shap_raw = self.explainer.shap_values(
            x_arr.reshape(1, -1),
            nsamples=nsamples,
            silent=True
        )

        # Handle different SHAP output formats (1D array or list)
        if isinstance(shap_raw, list):
            # If returned for multiple classes, index 0 corresponds to Malignant
            shap_vec = np.asarray(shap_raw[0]).ravel()
        else:
            shap_vec = np.asarray(shap_raw).ravel()

        shap_vals = [round(float(v), 5) for v in shap_vec]

        # 3. Categorize positive vs negative contributors
        positive_drivers = []
        negative_drivers = []

        for name, val, s_val in zip(self.feature_names, x_arr, shap_vals):
            entry = {
                "feature": name,
                "value": round(float(val), 4),
                "shap_value": s_val,
                "abs_shap": abs(s_val),
            }
            if s_val > 0:
                positive_drivers.append(entry)
            else:
                negative_drivers.append(entry)

        # Sort drivers by absolute magnitude
        positive_drivers.sort(key=lambda x: x["abs_shap"], reverse=True)
        negative_drivers.sort(key=lambda x: x["abs_shap"], reverse=True)

        risk_level = "HIGH_RISK_MALIGNANT" if p_malignant >= 0.5 else "LOW_RISK_BENIGN"

        return LocalExplanation(
            sample_id=sample_id,
            predicted_class=pred_class,
            class_label=class_label,
            malignant_probability=round(p_malignant, 4),
            benign_probability=round(p_benign, 4),
            feature_names=self.feature_names,
            feature_values=[round(float(v), 4) for v in x_arr],
            shap_values=shap_vals,
            base_value=round(self.base_value, 4),
            top_positive_features=positive_drivers[:5],
            top_negative_features=negative_drivers[:5],
            risk_assessment=risk_level,
            explanation_method="KernelExplainer",
            model_name=self.model_name,
            metadata={
                "background_size": len(self.background),
                "nsamples": nsamples,
            }
        )

    def explain_dataset(
        self,
        X_eval: Union[np.ndarray, pd.DataFrame],
        max_samples: Optional[int] = 50,
        nsamples: int = 80,
    ) -> Tuple[np.ndarray, List[GlobalFeatureImportance]]:
        """
        Computes SHAP values over an evaluation cohort to determine global feature ranking.
        
        Returns:
            Tuple of (shap_matrix of shape (N, D), list of GlobalFeatureImportance).
        """
        X_arr = np.asarray(X_eval, dtype=np.float64)
        if max_samples and len(X_arr) > max_samples:
            X_eval_sub = X_arr[:max_samples]
        else:
            X_eval_sub = X_arr

        logger.info(f"[SHAP] Computing global explanations for {len(X_eval_sub)} cohort samples...")
        shap_raw = self.explainer.shap_values(
            X_eval_sub,
            nsamples=nsamples,
            silent=True
        )

        if isinstance(shap_raw, list):
            shap_matrix = np.asarray(shap_raw[0])
        else:
            shap_matrix = np.asarray(shap_raw)

        # Global aggregations
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        mean_signed = np.mean(shap_matrix, axis=0)
        std_val = np.std(shap_matrix, axis=0)

        # Rank features descending by mean absolute attribution
        sorted_indices = np.argsort(mean_abs)[::-1]

        rankings = []
        for rank, idx in enumerate(sorted_indices, start=1):
            pos_ratio = np.mean(shap_matrix[:, idx] > 0)
            if pos_ratio > 0.65:
                trend = "positive_risk"
            elif pos_ratio < 0.35:
                trend = "protective"
            else:
                trend = "mixed"

            rankings.append(GlobalFeatureImportance(
                feature_name=self.feature_names[idx],
                feature_index=int(idx),
                mean_abs_shap=round(float(mean_abs[idx]), 5),
                mean_shap=round(float(mean_signed[idx]), 5),
                std_shap=round(float(std_val[idx]), 5),
                rank=rank,
                direction_trend=trend,
            ))

        return shap_matrix, rankings
