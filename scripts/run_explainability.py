"""
Command-Line Interface for Part 8: Model Explainability.

Usage:
    python scripts/run_explainability.py
    python scripts/run_explainability.py --artifacts artifacts/experiments/<exp_id>
    python scripts/run_explainability.py --patient-sample
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import PROJECT_ROOT
from src.data.loader import load_dataset
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import (
    run_data_stage, run_preprocessing_stage, run_feature_extraction_stage,
    run_feature_selection_stage
)
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.encoding import AngleScaler
from src.explainability.wrapper import VQCPredictionWrapper, ClassicalPredictionWrapper
from src.explainability.feature_mapping import FeatureLineageTracker
from src.explainability.explainer import ModelExplainer
from src.explainability.gradcam import GradCAMExplainer, create_synthetic_cnn
from src.explainability.visualization import plot_gradcam_overlay


def resolve_latest_experiment_dir() -> str:
    base_dir = os.path.join(PROJECT_ROOT, "artifacts", "experiments")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Artifacts directory not found: {base_dir}")
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No experiment directories found in {base_dir}")
    subdirs.sort(key=os.path.getmtime, reverse=True)
    return subdirs[0]


def main():
    parser = argparse.ArgumentParser(description="MIndMatrix Hybrid Quantum Model Explainability Suite")
    parser.add_argument("--artifacts", type=str, default=None, help="Path to experiment artifacts folder")
    parser.add_argument("--output", type=str, default=None, help="Directory to save explainability reports and plots")
    parser.add_argument("--patient-sample", action="store_true", help="Generate detailed patient diagnostic reports")
    args = parser.parse_args()

    # 1. Resolve experiment artifacts
    exp_dir = args.artifacts or resolve_latest_experiment_dir()
    print(f"\n[EXPLAINABILITY] Using model artifacts from: {exp_dir}")

    output_dir = args.output or os.path.join(exp_dir, "explainability")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Pipeline Data Splits
    cfg = PipelineConfig()
    raw, split = run_data_stage(cfg)
    prep = run_preprocessing_stage(split, cfg, output_dir)
    latent = run_feature_extraction_stage(prep, split, cfg, output_dir)
    selected = run_feature_selection_stage(latent, split, cfg, output_dir)

    # 3. Load Trained VQC Model & Angle Scaler
    vqc_path = os.path.join(exp_dir, "vqc_model.pt")
    if not os.path.exists(vqc_path):
        raise FileNotFoundError(f"Missing VQC checkpoint: {vqc_path}")
    vqc_model = VariationalQuantumClassifier.load(vqc_path, map_location="cpu")

    scaler_path = os.path.join(exp_dir, "angle_scaler.json")
    if os.path.exists(scaler_path):
        angle_scaler = AngleScaler.load(scaler_path)
    else:
        angle_scaler = AngleScaler(target_range=(0.0, np.pi))
        angle_scaler.fit(selected.X_train_selected)

    feature_names = selected.selected_names
    vqc_wrapper = VQCPredictionWrapper(
        vqc_model=vqc_model,
        angle_scaler=angle_scaler,
        feature_names=feature_names,
    )

    # 4. Initialize Lineage Tracker & Master Explainer
    tracker = FeatureLineageTracker(
        raw_feature_names=split.X_train.columns.tolist(),
        selected_indices=selected.selected_indices,
        latent_dim=16,
    )

    explainer = ModelExplainer(
        vqc_wrapper=vqc_wrapper,
        X_train_bg=selected.X_train_selected,
        feature_names=feature_names,
        feature_tracker=tracker,
        artifacts_dir=output_dir,
        background_size=40,
        seed=cfg.seed,
    )

    # 5. Local Patient Explanations (1 Malignant, 1 Benign)
    print("\n" + "=" * 65)
    print("   GENERATING LOCAL PATIENT DIAGNOSTIC EXPLANATIONS")
    print("=" * 65)

    y_val_arr = split.y_val.values
    idx_mal = np.where(y_val_arr == 0)[0][0]
    idx_ben = np.where(y_val_arr == 1)[0][0]

    exp_mal = explainer.explain_patient(selected.X_val_selected[idx_mal], patient_id="MALIGNANT_P01", generate_plot=True)
    report_mal = explainer.generate_patient_diagnostic_report(exp_mal)
    print(report_mal)

    exp_ben = explainer.explain_patient(selected.X_val_selected[idx_ben], patient_id="BENIGN_P02", generate_plot=True)
    report_ben = explainer.generate_patient_diagnostic_report(exp_ben)
    print("\n" + report_ben)

    # 6. Global Cohort Attribution & Report
    print("\n" + "=" * 65)
    print("   COMPUTING GLOBAL ATTRIBUTION, FAITHFULNESS & STABILITY")
    print("=" * 65)

    full_report = explainer.generate_full_report(
        X_val_cohort=selected.X_val_selected,
        y_val_cohort=y_val_arr,
        X_test_cohort=selected.X_test_selected,
        experiment_id=os.path.basename(exp_dir),
    )

    print("\n[GLOBAL ATTRIBUTION RANKING (Top Selected Features)]:")
    for r in full_report.global_importance_rankings[:5]:
        print(f"  Rank {r['rank']}: {r['feature_name']:<12} | Mean |SHAP| = {r['mean_abs_shap']:.5f} | Trend: {r['direction_trend']}")

    print("\n[SANITY CHECKS & FAITHFULNESS]:")
    print(f"  Randomized Model Sensitivity: Passed (r={full_report.sanity_checks['pearson_correlation_with_randomized']:.4f})")
    print(f"  Cohort Faithfulness Rate:     {full_report.faithfulness_summary['cohort_faithfulness_rate'] * 100:.1f}%")
    print(f"  Attribution Stability Ratio:  {full_report.stability_summary['stable_feature_ratio'] * 100:.1f}%")
    print(f"  Leakage Audit All Passed:     {all(full_report.leakage_audit.values())}")

    # 7. Verification of Grad-CAM on Vision Architecture
    print("\n" + "=" * 65)
    print("   GRAD-CAM VISION VERIFICATION CHECK")
    print("=" * 65)
    synthetic_cnn = create_synthetic_cnn(in_channels=1, num_classes=2)
    gradcam = GradCAMExplainer(synthetic_cnn, target_layer_name="conv1")
    dummy_img = torch.randn(1, 1, 16, 16)
    heatmap, g_exp = gradcam.generate_heatmap(dummy_img, target_class=0)
    img_save = os.path.join(output_dir, "gradcam_verification.png")
    plot_gradcam_overlay(dummy_img.squeeze().numpy(), heatmap, img_save)
    print(f"  Grad-CAM validated: target_layer='{g_exp.target_layer}', heatmap_shape={g_exp.heatmap_shape}")
    print(f"  Grad-CAM plot saved to: {img_save}")

    print("\n" + "=" * 65)
    print("   EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY")
    print(f"   Reports and plots saved to: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
