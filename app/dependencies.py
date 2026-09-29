"""
app/dependencies.py
FastAPI dependency injection providers.
"""
from __future__ import annotations

import uuid
from fastapi import Request
from app.services.model_manager import model_manager, ModelManager
from app.services.inference_service import inference_service, InferenceService
from app.services.explanation_service import explanation_service, ExplanationService


def get_model_manager() -> ModelManager:
    """Provides singleton ModelManager instance."""
    return model_manager


def get_inference_service() -> InferenceService:
    """Provides InferenceService instance."""
    return inference_service


def get_explanation_service() -> ExplanationService:
    """Provides ExplanationService instance."""
    return explanation_service


def get_request_id(request: Request) -> str:
    """Retrieves or generates unique request correlation ID."""
    if hasattr(request.state, "request_id"):
        return request.state.request_id
    req_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
    request.state.request_id = req_id
    return req_id
