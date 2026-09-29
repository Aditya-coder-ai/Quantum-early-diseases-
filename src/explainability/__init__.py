"""
Part 8: Model Explainability Package.
Provides SHAP attribution, permutation baselines, multi-level feature lineage tracking,
Grad-CAM validation, faithfulness testing, and diagnostic patient reporting.
"""
from src.explainability.schemas import (
    LocalExplanation, GlobalFeatureImportance, FaithfulnessMetric,
    StabilityMetric, GradCAMExplanation, ExplanationReport
)
from src.explainability.router import DataModalityRouter, UnsupportedModalityError
from src.explainability.wrapper import VQCPredictionWrapper, ClassicalPredictionWrapper, EndToEndHybridWrapper
from src.explainability.shap_explainer import HybridSHAPExplainer
from src.explainability.feature_importance import PermutationImportanceExplainer, extract_classical_model_importance
from src.explainability.feature_mapping import FeatureLineageTracker
from src.explainability.gradcam import GradCAMExplainer, create_synthetic_cnn
from src.explainability.validation import (
    evaluate_faithfulness, run_randomized_model_sanity_check,
    evaluate_attribution_stability, perform_explainability_leakage_audit
)
from src.explainability.visualization import (
    plot_global_shap_bar, plot_local_waterfall,
    plot_qaoa_vs_shap_comparison, plot_faithfulness_comparison, plot_gradcam_overlay
)
from src.explainability.explainer import ModelExplainer

__all__ = [
    "LocalExplanation",
    "GlobalFeatureImportance",
    "FaithfulnessMetric",
    "StabilityMetric",
    "GradCAMExplanation",
    "ExplanationReport",
    "DataModalityRouter",
    "UnsupportedModalityError",
    "VQCPredictionWrapper",
    "ClassicalPredictionWrapper",
    "EndToEndHybridWrapper",
    "HybridSHAPExplainer",
    "PermutationImportanceExplainer",
    "extract_classical_model_importance",
    "FeatureLineageTracker",
    "GradCAMExplainer",
    "create_synthetic_cnn",
    "evaluate_faithfulness",
    "run_randomized_model_sanity_check",
    "evaluate_attribution_stability",
    "perform_explainability_leakage_audit",
    "plot_global_shap_bar",
    "plot_local_waterfall",
    "plot_qaoa_vs_shap_comparison",
    "plot_faithfulness_comparison",
    "plot_gradcam_overlay",
    "ModelExplainer",
]
