"""
Part 9: Classical vs Hybrid Comparison Package.
"""
from src.comparison.config import ComparisonConfig, ModelConfig
from src.comparison.metrics import compute_extended_metrics, tune_decision_threshold, ResourceProfiler
from src.comparison.fairness import FairnessAuditor, FairnessViolationError
from src.comparison.benchmark import ModelBenchmarkEngine
from src.comparison.ablations import AblationSuite
from src.comparison.statistical import aggregate_multiseed_metrics, run_paired_statistical_tests
from src.comparison.explainability_comparison import compare_classical_vs_hybrid_explanations
from src.comparison.visualization import ComparisonVisualizer
from src.comparison.reporter import ComparisonReporter
from src.comparison.runner import ComparisonRunner

__all__ = [
    "ComparisonConfig",
    "ModelConfig",
    "compute_extended_metrics",
    "tune_decision_threshold",
    "ResourceProfiler",
    "FairnessAuditor",
    "FairnessViolationError",
    "ModelBenchmarkEngine",
    "AblationSuite",
    "aggregate_multiseed_metrics",
    "run_paired_statistical_tests",
    "compare_classical_vs_hybrid_explanations",
    "ComparisonVisualizer",
    "ComparisonReporter",
    "ComparisonRunner",
]
