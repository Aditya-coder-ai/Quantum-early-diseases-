"""
app/schemas/common.py
Common response schemas for Health, Readiness, and Model Info endpoints.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Lightweight process health response."""
    status: str = Field("healthy", description="Application service status")
    service: str = Field(..., description="Service identifier")
    version: str = Field(..., description="Application semantic version")
    model_loaded: bool = Field(..., description="Indicates if ML/VQC model artifacts are in memory")
    timestamp: float = Field(..., description="UNIX epoch timestamp")


class ReadyResponse(BaseModel):
    """Deep readiness response verifying inference capability."""
    ready: bool = Field(..., description="True if service is ready to process inference requests")
    model_loaded: bool = Field(..., description="True if model is loaded")
    artifacts_valid: bool = Field(..., description="True if artifact checksums and dimensions match")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic readiness details")


class ModelInfoResponse(BaseModel):
    """Safe public metadata regarding the active diagnostic model."""
    model_id: str = Field(..., description="Unique model experiment identifier")
    model_version: str = Field(..., description="Model release version")
    model_type: str = Field(..., description="Architecture type (e.g. hybrid_vqc)")
    input_feature_count: int = Field(30, description="Raw clinical features accepted")
    selected_feature_count: int = Field(8, description="Compact features selected for quantum encoding")
    selected_feature_names: List[str] = Field(..., description="Names of selected latent features")
    qubits: int = Field(8, description="Quantum register width in qubits")
    circuit_depth: int = Field(2, description="Ansatz variational repetitions")
    trainable_parameters: int = Field(16, description="Quantum rotation parameter count")
    quantum_backend: str = Field(..., description="Simulation device or hardware backend")
    classification_threshold: float = Field(..., description="Decision boundary threshold")
    preprocessing_strategy: str = Field(..., description="Zero-leakage normalization summary")
    imbalance_strategy: str = Field(..., description="Oversampling technique applied to train split")
    validation_status: str = Field("VALIDATED", description="Integrity check status")
