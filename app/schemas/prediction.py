"""
app/schemas/prediction.py
Pydantic schemas for single and batch prediction requests and responses.
"""
from __future__ import annotations

from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np

from src.security.validation import WDBC_FEATURE_COUNT, WDBC_FEATURE_NAMES, MAX_BATCH_SIZE


class PredictionRequest(BaseModel):
    """
    Diagnostic inference request for a single patient record.
    Features can be provided either as:
    1. A list of 30 float values ordered according to canonical WDBC features.
    2. A dictionary mapping all 30 WDBC feature names to their numeric values.
    """
    features: Union[List[float], Dict[str, float]] = Field(
        ...,
        description="Patient clinical feature measurements (30 continuous variables)"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional decision threshold override. Defaults to calibrated model threshold."
    )
    sample_id: Optional[str] = Field(
        default=None,
        description="Optional pseudonymized client sample identifier (non-PHI)."
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Union[List[float], Dict[str, float]]) -> Union[List[float], Dict[str, float]]:
        if isinstance(v, list):
            if len(v) != WDBC_FEATURE_COUNT:
                raise ValueError(
                    f"Feature list must contain exactly {WDBC_FEATURE_COUNT} values, got {len(v)}"
                )
            for idx, val in enumerate(v):
                if val is None or not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Feature at index {idx} must be a finite numeric value (not NaN or Inf)")
        elif isinstance(v, dict):
            missing = [name for name in WDBC_FEATURE_NAMES if name not in v]
            if missing:
                raise ValueError(f"Missing required feature keys: {missing[:5]} (total {len(missing)} missing)")
            for key, val in v.items():
                if val is None or not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Feature '{key}' must be a finite numeric value (not NaN or Inf)")
        else:
            raise ValueError("Features must be a list of 30 numbers or a dictionary of 30 feature names")
        return v


class SinglePredictionResult(BaseModel):
    """Diagnostic prediction assessment for an individual patient record."""
    predicted_class: int = Field(..., description="0 = Malignant (Disease Positive), 1 = Benign (Disease Negative)")
    diagnosis_label: str = Field(..., description="'Malignant' or 'Benign'")
    malignant_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of malignancy")
    benign_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of benign diagnosis")
    oncology_risk_assessment: str = Field(..., description="'HIGH_RISK_MALIGNANT' or 'LOW_RISK_BENIGN'")
    decision_threshold_used: float = Field(..., description="Threshold applied for binary classification")


class PredictionResponse(BaseModel):
    """Full structured response for single-sample diagnostic inference."""
    success: bool = True
    request_id: str = Field(..., description="Unique traceability identifier")
    model: Dict[str, str] = Field(..., description="Active model identifier and version")
    prediction: SinglePredictionResult = Field(..., description="Diagnostic assessment outcome")
    metadata: Dict[str, Any] = Field(..., description="Inference latency and circuit metadata")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple patient records."""
    samples: List[Union[List[float], Dict[str, float]]] = Field(
        ...,
        description=f"List of patient feature records (maximum {MAX_BATCH_SIZE} per batch)"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional decision threshold override for entire batch"
    )

    @field_validator("samples")
    @classmethod
    def validate_batch_size(cls, v: List[Any]) -> List[Any]:
        if not v:
            raise ValueError("Batch samples list cannot be empty")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum allowable limit of {MAX_BATCH_SIZE}")
        return v


class BatchPredictionResponse(BaseModel):
    """Structured response for batch diagnostic inference."""
    success: bool = True
    request_id: str = Field(..., description="Unique traceability identifier")
    count: int = Field(..., description="Number of samples evaluated in batch")
    predictions: List[SinglePredictionResult] = Field(..., description="Individual diagnostic assessments")
    total_inference_time_s: float = Field(..., description="Total batch inference duration in seconds")
    latency_per_sample_ms: float = Field(..., description="Average inference latency per sample in milliseconds")
    model: Dict[str, str] = Field(..., description="Model identifier and version")
