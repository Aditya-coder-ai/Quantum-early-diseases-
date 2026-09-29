"""
Master Command-Line Entrypoint for Part 7: Hybrid Classical-Quantum Pipeline.

Usage:
    python scripts/run_hybrid_pipeline.py --config configs/hybrid_pipeline.yaml
    python scripts/run_hybrid_pipeline.py --mode comparative
"""
from __future__ import annotations

import os
import sys
import argparse
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import PROJECT_ROOT
from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import HybridPipeline


def run_single_pipeline(config_path: str):
    """Run a single end-to-end hybrid pipeline configuration."""
    if os.path.exists(config_path):
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        print(f"Warning: {config_path} not found. Using default PipelineConfig.")
        cfg = PipelineConfig()

    pipeline = HybridPipeline(cfg)
    report = pipeline.run()
    return report


def run_comparative_experiment():
    """Run standardized comparative configurations A, B, C, D across identical splits."""
    print("\n" + "=" * 75)
    print("   RUNNING STANDARDIZED COMPARATIVE HYBRID PIPELINE BENCHMARKS")
    print("=" * 75)

    configs = [
        ("Config_A_Classical_FS_Classical_SVM", PipelineConfig(
            experiment_name="config_a_classical",
            feature_selection_method="mutual_info",
            imbalance_method="class_weight",
            vqc_epochs=15,
            vqc_patience=5,
        )),
        ("Config_B_QAOA_FS_VQC_Original", PipelineConfig(
            experiment_name="config_b_qaoa_vqc_orig",
            feature_selection_method="qaoa",
            imbalance_method="none",
            vqc_epochs=15,
            vqc_patience=5,
        )),
        ("Config_C_QAOA_FS_VQC_SMOTE", PipelineConfig(
            experiment_name="config_c_qaoa_vqc_smote",
            feature_selection_method="qaoa",
            imbalance_method="smote",
            vqc_epochs=15,
            vqc_patience=5,
        )),
        ("Config_D_QAOA_FS_VQC_QGAN", PipelineConfig(
            experiment_name="config_d_qaoa_vqc_qgan",
            feature_selection_method="qaoa",
            imbalance_method="qgan",
            vqc_epochs=15,
            vqc_patience=5,
        )),
    ]

    leaderboard = []

    for name, cfg in configs:
        print(f"\n>>> Running Benchmark: {name} <<<")
        pipeline = HybridPipeline(cfg)
        rep = pipeline.run()

        row = {
            "configuration": name,
            "experiment_id": rep.experiment_id,
            "feature_selection": cfg.feature_selection_method,
            "imbalance_method": cfg.imbalance_method,
            "val_accuracy": rep.validation_metrics["accuracy"],
            "val_f1": rep.validation_metrics["f1"],
            "val_pr_auc": rep.validation_metrics["pr_auc"],
            "val_minority_recall": rep.validation_metrics["minority_recall"],
            "test_accuracy": rep.test_metrics["accuracy"],
            "test_f1": rep.test_metrics["f1"],
            "test_roc_auc": rep.test_metrics["roc_auc"],
            "test_pr_auc": rep.test_metrics["pr_auc"],
            "test_minority_recall": rep.test_metrics["minority_recall"],
            "false_negatives": rep.test_metrics["malignant_false_negatives"],
            "total_runtime_s": rep.total_runtime_s,
        }
        leaderboard.append(row)

    lead_df = pd.DataFrame(leaderboard)
    out_csv = os.path.join(PROJECT_ROOT, "results", "hybrid_pipeline_comparative_table.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    lead_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 75)
    print("   COMPARATIVE BENCHMARK LEADERBOARD (FINAL TEST SET EVALUATED ONCE)")
    print("=" * 75)
    print(lead_df[["configuration", "test_accuracy", "test_f1", "test_roc_auc", "test_minority_recall", "false_negatives", "total_runtime_s"]].to_string(index=False))
    print(f"\nSaved consolidated benchmark to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run MIndMatrix Complete Hybrid Pipeline")
    parser.add_argument("--config", type=str, default="configs/hybrid_pipeline.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--mode", type=str, choices=["single", "comparative"], default="single", help="Execution mode")
    args = parser.parse_args()

    if args.mode == "comparative":
        run_comparative_experiment()
    else:
        run_single_pipeline(args.config)


if __name__ == "__main__":
    main()
