"""
Explainability Comparison between Classical Reference Model and Hybrid VQC for Part 9.
Reuses Part 8 Kernel SHAP explainers to quantify feature importance alignment and divergence.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Tuple
from sklearn.svm import SVC

from src.explainability.wrapper import ClassicalPredictionWrapper, VQCPredictionWrapper
from src.explainability.shap_explainer import HybridSHAPExplainer
from src.vqc.encoding import AngleScaler
from src.vqc.model import VariationalQuantumClassifier


def compare_classical_vs_hybrid_explanations(
    classical_model: SVC,
    hybrid_vqc: VariationalQuantumClassifier,
    angle_scaler: AngleScaler,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    n_background: int = 30,
    n_explain_samples: int = 20,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes a side-by-side Kernel SHAP comparison on identical test samples.
    Measures rank correlation, top features, and attribution directionality.
    """
    rng = np.random.RandomState(seed)
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    exp_idx = rng.choice(len(X_test), size=min(n_explain_samples, len(X_test)), replace=False)
    X_exp = X_test[exp_idx]

    # 1. Classical SHAP
    classical_wrapper = ClassicalPredictionWrapper(classical_model)
    classical_explainer = HybridSHAPExplainer(
        predict_fn=classical_wrapper.predict_malignant_proba,
        X_background=X_bg,
        feature_names=feature_names,
        background_size=len(X_bg),
        model_name="Classical_SVM",
        seed=seed
    )
    _, classical_rankings = classical_explainer.explain_dataset(X_exp, max_samples=len(X_exp), nsamples=60)

    # 2. Hybrid VQC SHAP
    vqc_wrapper = VQCPredictionWrapper(hybrid_vqc, angle_scaler)
    vqc_explainer = HybridSHAPExplainer(
        predict_fn=vqc_wrapper.predict_malignant_proba,
        X_background=X_bg,
        feature_names=feature_names,
        background_size=len(X_bg),
        model_name="Hybrid_VQC",
        seed=seed
    )
    _, vqc_rankings = vqc_explainer.explain_dataset(X_exp, max_samples=len(X_exp), nsamples=60)

    # Align features into comparison table
    cls_dict = {f.feature_name: f for f in classical_rankings}
    vqc_dict = {f.feature_name: f for f in vqc_rankings}

    comparison_rows = []
    cls_ranks = []
    vqc_ranks = []

    for name in feature_names:
        c_item = cls_dict.get(name)
        v_item = vqc_dict.get(name)

        c_abs = c_item.mean_abs_shap if c_item else 0.0
        c_rank = c_item.rank if c_item else 999
        c_mean = c_item.mean_shap if c_item else 0.0

        v_abs = v_item.mean_abs_shap if v_item else 0.0
        v_rank = v_item.rank if v_item else 999
        v_mean = v_item.mean_shap if v_item else 0.0

        cls_ranks.append(c_rank)
        vqc_ranks.append(v_rank)

        same_direction = (c_mean * v_mean) >= 0

        comparison_rows.append({
            "feature_name": name,
            "classical_mean_abs_shap": round(c_abs, 5),
            "classical_rank": c_rank,
            "classical_mean_shap": round(c_mean, 5),
            "vqc_mean_abs_shap": round(v_abs, 5),
            "vqc_rank": v_rank,
            "vqc_mean_shap": round(v_mean, 5),
            "same_direction": bool(same_direction),
        })

    # Sort by VQC importance
    comparison_rows.sort(key=lambda x: x["vqc_mean_abs_shap"], reverse=True)

    # Spearman rank correlation between classical and quantum attribution ranks
    spearman_rho, spearman_pval = stats.spearmanr(cls_ranks, vqc_ranks)

    return {
        "spearman_rank_correlation": round(float(spearman_rho), 4) if not np.isnan(spearman_rho) else 0.0,
        "spearman_p_value": round(float(spearman_pval), 6) if not np.isnan(spearman_pval) else 1.0,
        "n_samples_explained": len(X_exp),
        "comparison_table": comparison_rows,
        "top_classical_feature": classical_rankings[0].feature_name if classical_rankings else "N/A",
        "top_vqc_feature": vqc_rankings[0].feature_name if vqc_rankings else "N/A",
    }
