"""
app/api/health.py
Health and Readiness check endpoints.
"""
from __future__ import annotations

import time
from fastapi import APIRouter, Depends, Response, status

from app.config import settings
from app.schemas.common import HealthResponse, ReadyResponse
from app.dependencies import get_model_manager
from app.services.model_manager import ModelManager

router = APIRouter(tags=["Health & Readiness"])


@router.get("/health", response_model=HealthResponse, summary="Process Health Check")
def health_check(manager: ModelManager = Depends(get_model_manager)) -> HealthResponse:
    """Lightweight check verifying the web server process is alive and accepting traffic."""
    return HealthResponse(
        status="healthy",
        service="mindmatrix-disease-detection-inference",
        version=settings.app_version,
        model_loaded=manager.is_loaded,
        timestamp=time.time()
    )


@router.get("/ready", response_model=ReadyResponse, summary="Inference Readiness Check")
def readiness_check(
    response: Response,
    manager: ModelManager = Depends(get_model_manager)
) -> ReadyResponse:
    """
    Deep readiness check verifying that model checkpoints, preprocessors,
    and quantum scalers are loaded in memory and capable of performing predictions.
    """
    is_ready = manager.is_loaded and manager.artifacts_valid
    
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            ready=False,
            model_loaded=manager.is_loaded,
            artifacts_valid=manager.artifacts_valid,
            details={"error": manager.load_error or "Model components not yet loaded"}
        )

    return ReadyResponse(
        ready=True,
        model_loaded=True,
        artifacts_valid=True,
        details={
            "model_id": manager.model_id,
            "qubits": manager.vqc_model.n_qubits if manager.vqc_model else 0,
            "backend": settings.quantum_backend
        }
    )
