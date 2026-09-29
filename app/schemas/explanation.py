"""
app/schemas/explanation.py
Pydantic schemas for model explanation requests and responses.
"""
from __future__ import annotations

from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel, Field, field_validator

from src.security.validation import WDBC_FEATURE_COUNT, WDBC_FEATURE_NAMES
from app.schemas.prediction import SinglePredictionResult


class ExplanationRequest(BaseModel):
    """Request for local feature attribution on a patient diagnostic assessment."""
    features: Union[List[float], Dict[str, float]] = Field(
        ...,
        description="Patient clinical feature measurements (30 continuous variables)"
    )
    explanation_method: str = Field(
        default="shap",
        description="Interpretation method: 'shap' (Kernel SHAP) supported"
    )
    nsamples: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Number of Monte Carlo background permutations for Kernel SHAP (capped at 100 for latency)"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Decision threshold"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Union[List[float], Dict[str, float]]) -> Union[List[float], Dict[str, float]]:
        if isinstance(v, list) and len(v) != WDBC_FEATURE_COUNT:
            raise ValueError(f"Feature list must contain exactly {WDBC_FEATURE_COUNT} values, got {len(v)}")
        elif isinstance(v, dict):
            missing = [name for name in WDBC_FEATURE_NAMES if name not in v]
            if missing:
                raise ValueError(f"Missing required feature keys: {missing[:5]}")
        return v


class FeatureContribution(BaseModel):
    """Attribution score for a specific feature."""
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Observed feature value for this patient")
    shap_value: float = Field(..., description="SHAP attribution score (positive pushes toward malignancy)")
    abs_shap: float = Field(..., description="Magnitude of attribution importance")
    impact_direction: str = Field(..., description="'malignancy_driver' or 'protective_factor'")


class ExplanationData(BaseModel):
    """Detailed SHAP attribution details."""
    method: str = Field("SHAP (KernelExplainer)", description="Attribution algorithm")
    base_value: float = Field(..., description="Expected baseline malignancy probability across reference cohort")
    top_positive_drivers: List[FeatureContribution] = Field(..., description="Features increasing malignancy risk")
    top_negative_drivers: List[FeatureContribution] = Field(..., description="Features decreasing malignancy risk")
    all_contributions: List[FeatureContribution] = Field(..., description="All evaluated features ordered by absolute importance")
    clinical_caveat: str = Field(
        "Attribution reflects mathematical model sensitivity and does not prove biological causation.",
        description="Regulatory clinical notice"
    )


class ExplanationResponse(BaseModel):
    """Structured response for explanation requests."""
    success: bool = True
    request_id: str = Field(..., description="Unique traceability identifier")
    model: Dict[str, str] = Field(..., description="Model identifier and version")
    prediction: SinglePredictionResult = Field(..., description="Diagnostic assessment outcome")
    explanation: ExplanationData = Field(..., description="Local feature attribution scores")
    explanation_time_s: float = Field(..., description="Explanation computation duration in seconds")


class ImageExplanationResponse(BaseModel):
    """Response for image upload explanation requests."""
    success: bool
    request_id: str
    modality_status: str
    message: str
    details: Dict[str, Any]
