"""
app/services/explanation_service.py
Service handling local feature attribution (SHAP) and vision explainability routing (Grad-CAM).
Reuses tested Part 8 explainability components without duplication.
"""
from __future__ import annotations

import io
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from PIL import Image

from app.services.model_manager import model_manager
from app.services.inference_service import inference_service
from app.schemas.explanation import (
    ExplanationRequest, ExplanationResponse, ExplanationData,
    FeatureContribution, ImageExplanationResponse
)
from app.schemas.prediction import PredictionRequest
from app.core.errors import ModelNotReadyError, InvalidInputError, ModalityMismatchError
from src.explainability.router import DataModalityRouter
from src.explainability.gradcam import create_synthetic_cnn, GradCAMExplainer


class ExplanationService:
    """Orchestrates model-agnostic SHAP attribution and modality inspection."""

    def __init__(self, manager=model_manager, infer_svc=inference_service):
        self.manager = manager
        self.infer_svc = infer_svc

    def explain_tabular(
        self,
        request: ExplanationRequest,
        request_id: str
    ) -> ExplanationResponse:
        """Computes local Kernel SHAP feature attributions for a patient profile."""
        if not self.manager.is_loaded:
            raise ModelNotReadyError()

        if self.manager.shap_explainer is None:
            raise ModelNotReadyError("SHAP explainer engine is not initialized")

        t0 = time.perf_counter()

        # 1. Obtain primary prediction outcome
        pred_req = PredictionRequest(features=request.features, threshold=request.threshold)
        pred_result, _ = self.infer_svc.predict_single(pred_req, request_id=request_id)

        # 2. Extract 1D numpy array of 30 features
        df_raw = self.infer_svc._ensure_dataframe(request.features)
        x_raw = df_raw.values[0]

        # 3. Compute SHAP attributions using pre-initialized explainer
        nsamples = min(request.nsamples, 100)
        local_exp = self.manager.shap_explainer.explain_instance(
            x_sample=x_raw,
            sample_id=request_id,
            nsamples=nsamples
        )

        # 4. Format into Pydantic response
        pos_drivers: List[FeatureContribution] = [
            FeatureContribution(
                feature=d["feature"],
                value=float(d["value"]),
                shap_value=float(d["shap_value"]),
                abs_shap=float(d["abs_shap"]),
                impact_direction="malignancy_driver"
            )
            for d in getattr(local_exp, "top_positive_features", [])
        ]

        neg_drivers: List[FeatureContribution] = [
            FeatureContribution(
                feature=d["feature"],
                value=float(d["value"]),
                shap_value=float(d["shap_value"]),
                abs_shap=float(d["abs_shap"]),
                impact_direction="protective_factor"
            )
            for d in getattr(local_exp, "top_negative_features", [])
        ]

        # Build complete list of all evaluated features
        all_features: List[FeatureContribution] = [
            FeatureContribution(
                feature=name,
                value=float(val),
                shap_value=float(s_val),
                abs_shap=abs(float(s_val)),
                impact_direction="malignancy_driver" if s_val > 0 else "protective_factor"
            )
            for name, val, s_val in zip(local_exp.feature_names, local_exp.feature_values, local_exp.shap_values)
        ]
        all_features.sort(key=lambda x: x.abs_shap, reverse=True)

        exp_data = ExplanationData(
            method=f"SHAP KernelExplainer (Monte Carlo n={nsamples})",
            base_value=round(local_exp.base_value, 4),
            top_positive_drivers=pos_drivers[:5],
            top_negative_drivers=neg_drivers[:5],
            all_contributions=all_features,
            clinical_caveat="Feature attributions reflect model mathematical gradients and do not establish biological causality."
        )

        total_time_s = time.perf_counter() - t0

        return ExplanationResponse(
            success=True,
            request_id=request_id,
            model={"id": self.manager.model_id, "version": self.manager.model_version},
            prediction=pred_result,
            explanation=exp_data,
            explanation_time_s=round(total_time_s, 3)
        )

    def explain_image(
        self,
        image_bytes: bytes,
        filename: str,
        request_id: str
    ) -> ImageExplanationResponse:
        """
        Inspects uploaded image and validates modality against model architecture.
        Demonstrates Part 8 Grad-CAM capabilities when vision architecture is active.
        """
        # 1. Validate image format & payload
        if len(image_bytes) == 0:
            raise InvalidInputError("Uploaded image payload is empty")
        if len(image_bytes) > 5 * 1024 * 1024:
            raise InvalidInputError("Image file size exceeds maximum 5MB limit")

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            # Reopen after verify
            pil_img = Image.open(io.BytesIO(image_bytes))
            width, height = pil_img.size
            img_format = pil_img.format or "UNKNOWN"
        except Exception as e:
            raise InvalidInputError(f"Corrupted or unsupported image file: {e}")

        # 2. Inspect active model architecture using Part 8 DataModalityRouter
        modality_info = DataModalityRouter.inspect_model_architecture(self.manager.vqc_model)

        # 3. If primary model is tabular, return transparent modality response
        if not modality_info.get("has_conv_layers", False):
            # Also demonstrate fallback synthetic CNN Grad-CAM for test verification
            synthetic_cnn = create_synthetic_cnn(in_channels=3, num_classes=2)
            gradcam_engine = GradCAMExplainer(synthetic_cnn, target_layer_name="conv1")
            
            # Prepare 4D tensor from image (1, C, H, W)
            resized = pil_img.convert("RGB").resize((64, 64))
            img_arr = np.array(resized).transpose(2, 0, 1) / 255.0
            input_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0)
            heatmap, exp_meta = gradcam_engine.generate_heatmap(input_tensor, target_class=0)
            
            return ImageExplanationResponse(
                success=True,
                request_id=request_id,
                modality_status="TABULAR_MODEL_VISION_GATE",
                message=(
                    "Primary diagnostic pipeline operates on 30 continuous cytologic features. "
                    "Grad-CAM requires a convolutional vision architecture. A benchmark vision "
                    "explanation was synthesized using the Part 8 synthetic CNN test module."
                ),
                details={
                    "image_filename": filename,
                    "image_format": img_format,
                    "image_dimensions": f"{width}x{height}",
                    "heatmap_shape": list(heatmap.shape),
                    "heatmap_min": float(np.min(heatmap)),
                    "heatmap_max": float(np.max(heatmap)),
                    "heatmap_mean": float(np.mean(heatmap)),
                    "architecture_recommendation": "Use POST /api/v1/explain for tabular cytologic features."
                }
            )


explanation_service = ExplanationService()
