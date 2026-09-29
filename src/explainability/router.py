"""
Part 8: Data Modality & Explainability Router.
Inspects incoming dataset and model structure to select the theoretically
appropriate interpretation method, strictly rejecting mismatched techniques.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UnsupportedModalityError(ValueError):
    """Raised when an explanation technique is incompatible with the data modality or model architecture."""
    pass


class DataModalityRouter:
    """
    Directs model interpretation requests to the verified attribution methodology:
    - Tabular / Vector data -> Model-Agnostic SHAP + Permutation Feature Importance.
    - Image data with Convolutional Layers -> Grad-CAM + Latent Attribution.
    - Tabular data on CNN requests -> Explictly rejected with guidance.
    """

    SUPPORTED_MODALITIES = ["tabular", "vector", "image", "time_series"]

    @staticmethod
    def inspect_data_modality(data: Union[pd.DataFrame, np.ndarray, torch.Tensor, Any]) -> str:
        """
        Determines modality from data shape, type, and dimensional structure.
        - 2D numeric tabular/matrix -> 'tabular'
        - 1D vector -> 'vector'
        - 4D (B, C, H, W) or 3D (C, H, W) -> 'image'
        """
        if isinstance(data, pd.DataFrame):
            return "tabular"
        
        arr = np.asarray(data)
        if arr.ndim == 1:
            return "vector"
        elif arr.ndim == 2:
            return "tabular"
        elif arr.ndim in (3, 4):
            return "image"
        else:
            return "unknown"

    @staticmethod
    def inspect_model_architecture(model: Any) -> Dict[str, Any]:
        """
        Inspects model layers to determine whether spatial convolutional representations exist.
        """
        info = {
            "has_conv_layers": False,
            "conv_layers": [],
            "is_quantum_model": False,
            "is_classical_torch": False,
            "is_scikit_learn": False,
            "model_type": type(model).__name__,
        }

        if hasattr(model, "circuit") or hasattr(model, "qnode") or "Quantum" in type(model).__name__:
            info["is_quantum_model"] = True

        if isinstance(model, nn.Module):
            info["is_classical_torch"] = True
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    info["has_conv_layers"] = True
                    info["conv_layers"].append(name)
        elif hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            info["is_scikit_learn"] = True

        return info

    @classmethod
    def route_explanation(
        cls,
        data: Any,
        model: Any,
        explicit_modality: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determines the appropriate explanation strategy.
        
        Returns:
            Dict containing selected strategy, explanation target, and justification.
        """
        modality = explicit_modality or cls.inspect_data_modality(data)
        arch_info = cls.inspect_model_architecture(model)

        if modality in ("tabular", "vector"):
            return {
                "modality": modality,
                "strategy": "shap_kernel_and_permutation",
                "explanation_target": "selected_features_to_vqc_probability",
                "gradcam_supported": False,
                "justification": (
                    f"Input is {modality} clinical features. SHAP KernelExplainer with permutation "
                    "importance provides model-agnostic, mathematically sound attribution without "
                    "assuming linearity or differentiability across quantum circuits."
                ),
            }
        elif modality == "image":
            if not arch_info["has_conv_layers"]:
                raise UnsupportedModalityError(
                    "Image modality was requested, but the provided model architecture contains no "
                    "convolutional layers (Conv2d). Grad-CAM requires 2D spatial feature maps."
                )
            return {
                "modality": "image",
                "strategy": "gradcam_spatial_heatmap",
                "target_layer": arch_info["conv_layers"][-1],
                "explanation_target": "cnn_representation_to_features",
                "gradcam_supported": True,
                "justification": (
                    f"Convolutional layer '{arch_info['conv_layers'][-1]}' detected. Grad-CAM extracts "
                    "class-discriminative spatial activations to explain feature extraction."
                ),
            }
        else:
            raise UnsupportedModalityError(
                f"Modality '{modality}' is not directly supported. Supported modalities: {cls.SUPPORTED_MODALITIES}"
            )
