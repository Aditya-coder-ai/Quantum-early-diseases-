"""
Command-Line Interface for Standalone Hybrid Quantum Inference.

Usage:
    python scripts/predict.py --artifacts artifacts/experiments/<exp_id> --sample-patient
    python scripts/predict.py --artifacts artifacts/experiments/<exp_id> --input data/patients.csv --output results/predictions.csv
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import PROJECT_ROOT
from src.data.loader import load_dataset
from src.inference.predict import MedicalInferenceEngine


def resolve_latest_artifact_dir(base_dir: str) -> str:
    """Finds the most recently modified experiment directory."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Artifacts base directory not found: {base_dir}")
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No experiment directories found in {base_dir}")
    subdirs.sort(key=os.path.getmtime, reverse=True)
    return subdirs[0]


def main():
    parser = argparse.ArgumentParser(description="MIndMatrix Standalone Hybrid Quantum Diagnostic Inference")
    parser.add_argument("--artifacts", type=str, default=None, help="Path to experiment artifacts folder")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV with raw clinical features")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction outputs CSV")
    parser.add_argument("--sample-patient", action="store_true", help="Run inference on test samples from dataset")
    args = parser.parse_args()

    # Resolve artifact directory
    if args.artifacts is None:
        base_dir = os.path.join(PROJECT_ROOT, "artifacts", "experiments")
        artifact_dir = resolve_latest_artifact_dir(base_dir)
        print(f"[INFERENCE] Using latest experiment directory: {artifact_dir}")
    else:
        artifact_dir = args.artifacts

    engine = MedicalInferenceEngine(artifact_dir)

    # If --sample-patient is specified or no input given, pick 2 samples (1 malignant, 1 benign) from dataset
    if args.sample_patient or args.input is None:
        print("[INFERENCE] Loading reference test patient samples...")
        X, y = load_dataset(save_raw=False)
        # Select 1 malignant (0) and 1 benign (1)
        idx_mal = y[y == 0].index[0]
        idx_ben = y[y == 1].index[0]
        sample_df = X.loc[[idx_mal, idx_ben]].copy()
        true_labels = ["Malignant (Class 0)", "Benign (Class 1)"]

        print(f"\n[INFERENCE] Running quantum prediction on {len(sample_df)} patient profiles...")
        res = engine.predict(sample_df)

        for i in range(len(sample_df)):
            print("\n" + "-" * 50)
            print(f"  Patient {i+1} Ground Truth:  {true_labels[i]}")
            print(f"  Predicted Class:       {res['predicted_classes'][i]} ({res['diagnosis_labels'][i]})")
            print(f"  Malignancy Probability: {res['malignant_probabilities'][i] * 100:.2f}%")
            print(f"  Benign Probability:     {res['benign_probabilities'][i] * 100:.2f}%")
            print(f"  Risk Assessment:        {res['oncology_risk_assessment'][i]}")
        print("-" * 50)
        print(f"  Average inference latency: {res['latency_per_sample_ms']:.2f} ms per patient")
        return

    # Load input from specified CSV file
    print(f"[INFERENCE] Loading input patient data from {args.input}...")
    input_df = pd.read_csv(args.input)
    res = engine.predict(input_df)

    res_df = input_df.copy()
    res_df["predicted_class"] = res["predicted_classes"]
    res_df["diagnosis_label"] = res["diagnosis_labels"]
    res_df["malignant_prob"] = res["malignant_probabilities"]
    res_df["risk_assessment"] = res["oncology_risk_assessment"]

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        res_df.to_csv(args.output, index=False)
        print(f"[INFERENCE] Predictions successfully written to: {args.output}")
    else:
        print(res_df[["diagnosis_label", "malignant_prob", "risk_assessment"]].to_string())


if __name__ == "__main__":
    main()
