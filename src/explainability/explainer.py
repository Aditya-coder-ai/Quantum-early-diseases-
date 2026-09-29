"""
Part 8: Master Model Explainability Orchestrator.
Coordinates local patient diagnostic explanations, global cohort attribution,
faithfulness verification, stability testing, and structured artifact generation.
"""
from __future__ import annotations

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from configs.config import PROJECT_ROOT
from src.explainability.schemas import (
    LocalExplanation, GlobalFeatureImportance, ExplanationReport, FaithfulnessMetric
)
from src.explainability.router import DataModalityRouter, UnsupportedModalityError
from src.explainability.wrapper import VQCPredictionWrapper, ClassicalPredictionWrapper, EndToEndHybridWrapper
from src.explainability.shap_explainer import HybridSHAPExplainer
from src.explainability.feature_importance import PermutationImportanceExplainer, extract_classical_model_importance
from src.explainability.feature_mapping import FeatureLineageTracker
from src.explainability.validation import (
    evaluate_faithfulness, run_randomized_model_sanity_check,
    evaluate_attribution_stability, perform_explainability_leakage_audit
)
from src.explainability.visualization import (
    plot_global_shap_bar, plot_local_waterfall,
    plot_qaoa_vs_shap_comparison, plot_faithfulness_comparison
)

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    High-level orchestrator providing unified, mathematically grounded explanations
    for hybrid classical-quantum medical diagnostic systems.
    """
    def __init__(
        self,
        vqc_wrapper: VQCPredictionWrapper,
        X_train_bg: np.ndarray,
        feature_names: List[str],
        classical_wrapper: Optional[ClassicalPredictionWrapper] = None,
        feature_tracker: Optional[FeatureLineageTracker] = None,
        artifacts_dir: Optional[str] = None,
        background_size: int = 40,
        seed: int = 42,
    ):
        self.vqc_wrapper = vqc_wrapper
        self.classical_wrapper = classical_wrapper
        self.feature_names = list(feature_names)
        self.artifacts_dir = artifacts_dir or os.path.join(PROJECT_ROOT, "artifacts", "explainability")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.seed = seed

        # Ensure background is drawn strictly from train
        self.X_train_bg = np.asarray(X_train_bg, dtype=np.float64)

        # Feature lineage tracker
        self.feature_tracker = feature_tracker or FeatureLineageTracker(
            raw_feature_names=[f"raw_{i}" for i in range(30)],
            selected_indices=list(range(len(feature_names))),
        )

        # Initialize primary SHAP explainer on VQC wrapper
        self.shap_explainer = HybridSHAPExplainer(
            predict_fn=self.vqc_wrapper.predict_malignant_proba,
            X_background=self.X_train_bg,
            feature_names=self.feature_names,
            background_size=background_size,
            model_name="Hybrid_VQC",
            seed=seed,
        )

        # Initialize permutation baseline explainer
        self.perm_explainer = PermutationImportanceExplainer(
            predict_fn=self.vqc_wrapper.predict_malignant_proba,
            metric="roc_auc",
            n_repeats=3,
            seed=seed,
        )

    def explain_patient(
        self,
        x_patient: Union[np.ndarray, pd.Series, List[float]],
        patient_id: Union[int, str] = "P001",
        generate_plot: bool = True,
    ) -> LocalExplanation:
        """
        Produces a local patient-specific attribution profile.
        """
        local_exp = self.shap_explainer.explain_instance(
            x_sample=x_patient,
            sample_id=patient_id,
            nsamples=100
        )

        if generate_plot:
            plot_path = os.path.join(self.artifacts_dir, f"patient_{patient_id}_waterfall.png")
            plot_local_waterfall(local_exp, plot_path)
            local_exp.metadata["plot_path"] = plot_path

        return local_exp

    def explain_cohort(
        self,
        X_cohort: np.ndarray,
        y_cohort: np.ndarray,
        max_samples: int = 40,
        generate_plots: bool = True,
    ) -> Tuple[np.ndarray, List[GlobalFeatureImportance], pd.DataFrame, List[FaithfulnessMetric]]:
        """
        Generates global feature attributions, contrasts QAOA vs SHAP,
        computes permutation baselines, and evaluates explanation faithfulness.
        """
        # 1. SHAP global dataset explanation
        shap_matrix, global_rankings = self.shap_explainer.explain_dataset(
            X_cohort, max_samples=max_samples, nsamples=80
        )

        # 2. Permutation feature importance baseline
        perm_importance = self.perm_explainer.compute_importance(
            X_cohort[:max_samples], y_cohort[:max_samples], self.feature_names
        )

        # 3. QAOA vs SHAP comparison
        shap_imp_dict = {r.feature_name: r.mean_abs_shap for r in global_rankings}
        comp_df = self.feature_tracker.build_qaoa_vs_shap_comparison(
            shap_importance=shap_imp_dict,
            classical_importance=perm_importance
        )

        # 4. Faithfulness / Perturbation checks across first 3 samples
        faithfulness_results = []
        for i in range(min(3, len(X_cohort))):
            fm = evaluate_faithfulness(
                predict_fn=self.vqc_wrapper.predict_malignant_proba,
                x_sample=X_cohort[i],
                shap_values=shap_matrix[i].tolist(),
                feature_names=self.feature_names,
                k=2,
                sample_id=i + 1
            )
            faithfulness_results.append(fm)

        # 5. Visualizations
        if generate_plots:
            plot_global_shap_bar(
                global_rankings,
                os.path.join(self.artifacts_dir, "global_feature_importance_bar.png"),
                title="VQC Global Feature Attribution (Mean |SHAP|)"
            )
            plot_qaoa_vs_shap_comparison(
                comp_df,
                os.path.join(self.artifacts_dir, "qaoa_vs_shap_comparison.png")
            )
            plot_faithfulness_comparison(
                faithfulness_results,
                os.path.join(self.artifacts_dir, "faithfulness_perturbation_check.png")
            )

        return shap_matrix, global_rankings, comp_df, faithfulness_results

    def generate_patient_diagnostic_report(self, local_exp: LocalExplanation) -> str:
        """
        Formats a clinically interpretable diagnostic summary for an oncologist.
        """
        lines = [
            "=" * 65,
            f"   ONCOLOGICAL DIAGNOSTIC EXPLAINABILITY REPORT — PATIENT {local_exp.sample_id}",
            "=" * 65,
            f"Model Prediction:       {local_exp.class_label.upper()} (Class {local_exp.predicted_class})",
            f"Malignancy Probability: {local_exp.malignant_probability * 100:.2f}%",
            f"Benign Probability:     {local_exp.benign_probability * 100:.2f}%",
            f"Population Base Rate:   {local_exp.base_value * 100:.2f}%",
            f"Assessed Risk Tier:     {local_exp.risk_assessment}",
            "-" * 65,
            "TOP RISK-INCREASING CONTRIBUTORS (Pushed toward Malignancy):",
        ]

        if local_exp.top_positive_features:
            for i, feat in enumerate(local_exp.top_positive_features, start=1):
                lines.append(f"  {i}. {feat['feature']:<12} = {feat['value']:<8.4f} (SHAP: +{feat['shap_value']:.4f})")
        else:
            lines.append("  None detected.")

        lines.append("-" * 65)
        lines.append("TOP PROTECTIVE CONTRIBUTORS (Pushed toward Benign):")
        if local_exp.top_negative_features:
            for i, feat in enumerate(local_exp.top_negative_features, start=1):
                lines.append(f"  {i}. {feat['feature']:<12} = {feat['value']:<8.4f} (SHAP: {feat['shap_value']:.4f})")
        else:
            lines.append("  None detected.")

        lines.extend([
            "=" * 65,
            "ETHICAL & METHODOLOGICAL DISCLAIMER:",
            "  This output is a model-attributed feature attribution (SHAP KernelExplainer)",
            "  reflecting mathematical sensitivities of the hybrid quantum classifier.",
            "  It does NOT constitute medical causation or definitive clinical diagnosis.",
            "  All findings must be confirmed by certified oncology personnel.",
            "=" * 65,
        ])
        return "\n".join(lines)

    def generate_full_report(
        self,
        X_val_cohort: np.ndarray,
        y_val_cohort: np.ndarray,
        X_test_cohort: np.ndarray,
        experiment_id: str = "exp_part8_explainability",
    ) -> ExplanationReport:
        """
        Executes end-to-end explainability suite, validates stability & leakage,
        and saves explainability_report.json and feature_importance.csv.
        """
        logger.info(f"[EXPLAINABILITY] Running full cohort explainability report for [{experiment_id}]...")

        shap_mat, rankings, comp_df, faith_metrics = self.explain_cohort(
            X_val_cohort, y_val_cohort, max_samples=40, generate_plots=True
        )

        # Stability Analysis
        stability_metrics = evaluate_attribution_stability(
            predict_fn=self.vqc_wrapper.predict_malignant_proba,
            X_sample=X_val_cohort[:15],
            X_train_bg=self.X_train_bg,
            feature_names=self.feature_names,
            n_runs=3,
            background_size=min(30, len(self.X_train_bg)),
        )

        # Leakage Audit
        leakage_audit = perform_explainability_leakage_audit(
            X_train_bg=self.X_train_bg,
            X_val_cohort=X_val_cohort,
            X_test_cohort=X_test_cohort,
            shap_values=shap_mat,
        )

        # Sanity check: randomized model
        # Create a dummy randomized predict function
        rand_weights = np.random.randn(len(self.feature_names))
        def rand_predict_fn(x):
            arr = np.asarray(x)
            logits = arr @ rand_weights
            return 1.0 / (1.0 + np.exp(-logits))

        sanity_res = run_randomized_model_sanity_check(
            trained_shap_values=shap_mat[:10],
            randomized_model_predict_fn=rand_predict_fn,
            X_sample=X_val_cohort[:10],
            X_background=self.X_train_bg,
            seed=self.seed
        )

        # Export feature_importance.csv
        csv_path = os.path.join(self.artifacts_dir, "feature_importance.csv")
        comp_df.to_csv(csv_path, index=False)
        results_csv = os.path.join(PROJECT_ROOT, "results", "feature_importance.csv")
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        comp_df.to_csv(results_csv, index=False)

        # Create report object
        report = ExplanationReport(
            experiment_id=experiment_id,
            data_modality="tabular_vector",
            explanation_target="compact_selected_features_to_vqc_probability",
            primary_explainer="SHAP_KernelExplainer",
            num_samples_explained=len(shap_mat),
            background_size=len(self.shap_explainer.background),
            global_importance_rankings=[r.to_dict() for r in rankings],
            qaoa_vs_shap_comparison=comp_df.to_dict(orient="records"),
            faithfulness_summary={
                "cohort_faithfulness_rate": round(float(np.mean([m.is_faithful for m in faith_metrics])), 4),
                "samples": [m.to_dict() for m in faith_metrics],
            },
            stability_summary={
                "mean_coefficient_of_variation": round(float(np.mean([s.coefficient_of_variation for s in stability_metrics])), 4),
                "stable_feature_ratio": round(float(np.mean([s.is_stable for s in stability_metrics])), 4),
                "features": [s.to_dict() for s in stability_metrics],
            },
            sanity_checks={
                "randomized_model_check_passed": sanity_res["passed"],
                "pearson_correlation_with_randomized": sanity_res["pearson_correlation"],
            },
            leakage_audit=leakage_audit,
            artifacts={
                "feature_importance_csv": csv_path,
                "global_bar_plot": os.path.join(self.artifacts_dir, "global_feature_importance_bar.png"),
                "qaoa_comparison_plot": os.path.join(self.artifacts_dir, "qaoa_vs_shap_comparison.png"),
                "faithfulness_plot": os.path.join(self.artifacts_dir, "faithfulness_perturbation_check.png"),
            }
        )

        # Export explainability_report.json
        json_path = os.path.join(self.artifacts_dir, "explainability_report.json")
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        results_json = os.path.join(PROJECT_ROOT, "results", "explainability_report.json")
        with open(results_json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report
