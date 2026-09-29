"""
Part 8: Explainability Schemas and Data Contracts.
Defines strongly typed dataclasses for local and global model explanations,
faithfulness metrics, stability analyses, and diagnostic reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union


@dataclass
class LocalExplanation:
    """Attribution analysis for a single patient / sample prediction."""
    sample_id: Union[int, str]
    predicted_class: int  # 0: Malignant, 1: Benign
    class_label: str       # 'Malignant' or 'Benign'
    malignant_probability: float
    benign_probability: float
    feature_names: List[str]
    feature_values: List[float]
    shap_values: List[float]  # Attribution to malignant disease probability
    base_value: float
    top_positive_features: List[Dict[str, Any]]  # Drives probability toward malignancy
    top_negative_features: List[Dict[str, Any]]  # Drives probability away from malignancy
    risk_assessment: str
    explanation_method: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalFeatureImportance:
    """Global feature attribution summary across an evaluation dataset."""
    feature_name: str
    feature_index: int
    mean_abs_shap: float
    mean_shap: float
    std_shap: float
    rank: int
    selected_by_qaoa: bool = False
    qaoa_rank: Optional[int] = None
    classical_importance: Optional[float] = None
    direction_trend: str = "neutral"  # 'positive_risk', 'protective', or 'mixed'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaithfulnessMetric:
    """Quantifies whether perturbing attributed features predictably shifts predictions."""
    sample_id: Union[int, str]
    original_probability: float
    top_k_features: List[str]
    top_k_perturbed_probability: float
    top_k_delta: float
    least_k_features: List[str]
    least_k_perturbed_probability: float
    least_k_delta: float
    is_faithful: bool  # True if |top_k_delta| > |least_k_delta|

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StabilityMetric:
    """Measures attribution reproducibility across different seeds or background samples."""
    feature_name: str
    mean_importance: float
    std_importance: float
    coefficient_of_variation: float
    is_stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GradCAMExplanation:
    """Attribution heatmap for convolutional feature extraction layers."""
    target_layer: str
    input_shape: Tuple[int, ...]
    heatmap_shape: Tuple[int, ...]
    heatmap_min: float
    heatmap_max: float
    overlay_path: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationReport:
    """Comprehensive summary of model explainability, faithfulness, and sanity checks."""
    experiment_id: str
    data_modality: str
    explanation_target: str
    primary_explainer: str
    num_samples_explained: int
    background_size: int
    global_importance_rankings: List[Dict[str, Any]]
    qaoa_vs_shap_comparison: List[Dict[str, Any]]
    faithfulness_summary: Dict[str, Any]
    stability_summary: Dict[str, Any]
    sanity_checks: Dict[str, bool]
    leakage_audit: Dict[str, bool]
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
