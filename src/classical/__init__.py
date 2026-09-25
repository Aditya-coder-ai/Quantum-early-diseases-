"""
Classical baseline models and benchmarking package.
"""
from src.classical.baseline import (
    run_classical_baselines,
    run_compact_feature_experiments,
    train_eval_classifier,
    get_class_imbalance_info
)

__all__ = [
    "run_classical_baselines",
    "run_compact_feature_experiments",
    "train_eval_classifier",
    "get_class_imbalance_info"
]
