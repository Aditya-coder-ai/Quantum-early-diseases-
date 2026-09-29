"""
app/api/model.py
Model information and metadata endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas.common import ModelInfoResponse
from app.dependencies import get_model_manager
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/model", tags=["Model Information"])


@router.get("/info", response_model=ModelInfoResponse, summary="Retrieve Active Model Metadata")
def get_model_info(manager: ModelManager = Depends(get_model_manager)) -> ModelInfoResponse:
    """
    Exposes safe, non-sensitive architectural information about the loaded hybrid model.
    Never exposes internal filesystem paths, training labels, or raw patient records.
    """
    metadata = manager.get_metadata()
    return ModelInfoResponse(**metadata)
