"""
app/core/errors.py
Structured exception classes and safe HTTP error formatters.
Enforces zero traceback or filesystem internal leakage in error responses.
"""
from __future__ import annotations

import time
from typing import Optional, Dict, Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class BaseAppException(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelNotReadyError(BaseAppException):
    """Raised when inference is requested before model artifacts are successfully loaded."""
    def __init__(self, message: str = "Inference service is initializing or artifacts not loaded"):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


class InvalidInputError(BaseAppException):
    """Raised when client input fails clinical feature schema or physiological bounds."""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)


class BatchSizeExceededError(BaseAppException):
    """Raised when a batch prediction request exceeds MAX_BATCH_SIZE."""
    def __init__(self, size: int, max_size: int):
        super().__init__(
            f"Batch size {size} exceeds maximum allowable limit of {max_size} samples",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class ModalityMismatchError(BaseAppException):
    """Raised when an explanation technique is incompatible with model architecture."""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)


class ArtifactNotFoundError(BaseAppException):
    """Raised when a required pipeline artifact cannot be located on disk."""
    def __init__(self, artifact_name: str):
        super().__init__(f"Required artifact '{artifact_name}' not found", status_code=status.HTTP_404_NOT_FOUND)


def format_error_response(
    request: Request,
    status_code: int,
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Produces a clean, standardized, privacy-preserving JSON error payload."""
    request_id = getattr(request.state, "request_id", "req_unknown")
    content = {
        "success": False,
        "request_id": request_id,
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
            "timestamp": time.time(),
        }
    }
    if details:
        content["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=content)


async def app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
    """Handles all subclasses of BaseAppException."""
    return format_error_response(
        request=request,
        status_code=exc.status_code,
        error_type=exc.__class__.__name__,
        message=exc.message
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handles Pydantic request body validation failures with clean field summaries."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        errors.append({"field": loc, "issue": err.get("msg", "Invalid value")})
    return format_error_response(
        request=request,
        status_code=422,
        error_type="ValidationError",
        message="Request schema validation failed. Please check input parameters.",
        details={"validation_errors": errors}
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handles unexpected server crashes safely without exposing stack traces."""
    # In production, never leak internal traceback strings to API clients
    return format_error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="InternalServerError",
        message="An unexpected server error occurred during inference processing."
    )
