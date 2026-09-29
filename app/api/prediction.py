"""
app/api/prediction.py
Single and batch diagnostic inference endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.schemas.prediction import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from app.dependencies import get_inference_service, get_request_id
from app.services.inference_service import InferenceService
from app.core.security import verify_auth_dependency

router = APIRouter(prefix="/predict", tags=["Diagnostic Inference"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Single Patient Diagnostic Inference",
    dependencies=[Depends(verify_auth_dependency)]
)
def predict_single_sample(
    request_body: PredictionRequest,
    raw_request: Request,
    service: InferenceService = Depends(get_inference_service),
    request_id: str = Depends(get_request_id),
) -> PredictionResponse:
    """
    Executes the full hybrid classical-quantum diagnostic pipeline on a single patient record:
    1. Validates 30 continuous cytologic features
    2. Applies zero-leakage normalization
    3. Extracts 16D classical latent representation
    4. Slices 8D QAOA-selected features
    5. Encodes into quantum angles on an 8-qubit register
    6. Evaluates Variational Quantum Classifier (VQC)
    7. Applies calibrated decision threshold
    8. Returns malignancy risk and oncology assessment
    """
    prediction_result, metadata = service.predict_single(request_body, request_id=request_id)
    
    return PredictionResponse(
        success=True,
        request_id=request_id,
        model={"id": service.manager.model_id, "version": service.manager.model_version},
        prediction=prediction_result,
        metadata=metadata
    )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Diagnostic Inference",
    dependencies=[Depends(verify_auth_dependency)]
)
def predict_batch_samples(
    request_body: BatchPredictionRequest,
    raw_request: Request,
    service: InferenceService = Depends(get_inference_service),
    request_id: str = Depends(get_request_id),
) -> BatchPredictionResponse:
    """
    Executes vectorized batch inference for up to 100 patient records simultaneously.
    Reduces redundant tensor conversions and leverages batch circuit broadcasting.
    """
    return service.predict_batch(request_body, request_id=request_id)
