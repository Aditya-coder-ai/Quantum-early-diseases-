"""
app/api/explanation.py
Local feature attribution (SHAP) and vision explainability (Grad-CAM) endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, UploadFile, File

from app.schemas.explanation import (
    ExplanationRequest, ExplanationResponse, ImageExplanationResponse
)
from app.dependencies import get_explanation_service, get_request_id
from app.services.explanation_service import ExplanationService
from app.core.security import verify_auth_dependency

router = APIRouter(prefix="/explain", tags=["Model Explainability"])


@router.post(
    "",
    response_model=ExplanationResponse,
    summary="Compute Local Feature Attribution (SHAP)",
    dependencies=[Depends(verify_auth_dependency)]
)
def explain_patient_prediction(
    request_body: ExplanationRequest,
    raw_request: Request,
    service: ExplanationService = Depends(get_explanation_service),
    request_id: str = Depends(get_request_id),
) -> ExplanationResponse:
    """
    Computes local feature attribution using Kernel SHAP:
    1. Evaluates model diagnostic prediction
    2. Measures Shapley attribution values against baseline training cohort
    3. Categorizes features into malignancy drivers vs protective factors
    4. Ranks all features by attribution magnitude
    """
    return service.explain_tabular(request_body, request_id=request_id)


@router.post(
    "/image",
    response_model=ImageExplanationResponse,
    summary="Inspect Modality and Synthesize Vision Explainability (Grad-CAM)",
    dependencies=[Depends(verify_auth_dependency)]
)
async def explain_image_upload(
    file: UploadFile = File(..., description="Medical specimen image (PNG, JPEG, TIFF)"),
    service: ExplanationService = Depends(get_explanation_service),
    request_id: str = Depends(get_request_id),
) -> ImageExplanationResponse:
    """
    Uploads a clinical image specimen to verify model architecture compatibility:
    - Inspects whether active model contains convolutional layers
    - Synthesizes Grad-CAM class activation map via Part 8 vision test module
    - Explains modality routing decisions
    """
    contents = await file.read()
    return service.explain_image(
        image_bytes=contents,
        filename=file.filename or "uploaded_specimen.png",
        request_id=request_id
    )
