"""
Master Orchestration Script for Part 3 — Classical Baseline + Classical Feature Extraction.

Executes end-to-end:
1. Verification of Part 2 preprocessed datasets
2. Training and evaluation of reference classical baseline models on raw 30 features
3. Feature extraction and dimensionality reduction across 4, 8, 12, and 16 dimensions (PCA & Autoencoder)
4. Feature quality verification and compact feature matrix export
5. Evaluation of classifiers on compact feature representations
6. Logging to experiment registry and saving trained model checkpoints
"""
import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import PROJECT_ROOT, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR
from src.features.extraction import run_feature_extraction_pipeline
from src.classical.baseline import run_classical_baselines, run_compact_feature_experiments


def main():
    print("\n" + "=" * 70)
    print("   PART 3 — CLASSICAL BASELINE + CLASSICAL FEATURE EXTRACTION")
    print("=" * 70)

    # 1. Verify Part 2 dataset availability
    req_files = ["X_train.csv", "X_val.csv", "X_test.csv", "y_train.csv", "y_val.csv", "y_test.csv"]
    for f in req_files:
        p = os.path.join(DATA_PROCESSED_DIR, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Part 2 processed file missing: {p}. Run scripts/run_preprocessing.py first.")

    print("\n[STEP 1/4] Part 2 Processed Dataset Verified.")

    # 2. Objective A: Classical Baseline Models on 30 Features
    print("\n[STEP 2/4] Executing Objective A: Classical Baseline Models...")
    baseline_results = run_classical_baselines(save_models=True)

    # 3. Objective B: Classical Feature Extraction (4, 8, 12, 16 Dims)
    print("\n[STEP 3/4] Executing Objective B: Feature Extraction (PCA & Autoencoder)...")
    extraction_results = run_feature_extraction_pipeline(target_dims=[4, 8, 12, 16])

    # 4. Compact Feature Classifier Evaluation
    print("\n[STEP 4/4] Evaluating Classifiers on Compact Features...")
    compact_results = run_compact_feature_experiments(target_dims=[4, 8, 12, 16])

    # 5. Export Part 3 Final Summary Report
    summary_file = os.path.join(RESULTS_DIR, "part3_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "status": "COMPLETE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baselines_count": len(baseline_results),
            "compact_experiments_count": len(compact_results),
            "extraction_methods": ["PCA", "Autoencoder"],
            "target_dimensions": [4, 8, 12, 16],
            "baseline_summary": baseline_results,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("   PART 3 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Summary Report: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
