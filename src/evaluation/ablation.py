"""
Ablation study and final comparison module.

Implements the five ablation experiments:
A) Raw features -> classical classifier
B) Raw features -> neural network (MLP)  
C) Raw features -> encoder -> 16 latent features -> classical classifier
D) Raw features -> encoder -> 16 latent -> classical feature selection -> classifier
E) Raw features -> encoder -> 16 latent -> quantum feature selection -> VQC

Generates final_comparison.csv and experiment logs.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, RESULTS_DIR, DATA_PROCESSED_DIR, SELECTED_DIM
)
from src.evaluation.metrics import compute_metrics, print_metrics, save_metrics


def load_all_data() -> dict:
    """Load all preprocessed and intermediate data."""
    data = {}
    
    # Raw preprocessed features
    for split in ["train", "val", "test"]:
        data[f"X_raw_{split}"] = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, f"X_{split}.csv")).values
        data[f"y_{split}"] = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")).values.ravel()
    
    # Latent features (16-dim)
    for split in ["train", "val", "test"]:
        path = os.path.join(RESULTS_DIR, f"latent_features_{split}.csv")
        if os.path.exists(path):
            data[f"X_latent_{split}"] = pd.read_csv(path).values
    
    # Selected features (8-dim, quantum)
    for split in ["train", "val", "test"]:
        path = os.path.join(RESULTS_DIR, f"selected_features_{split}.csv")
        if os.path.exists(path):
            data[f"X_selected_{split}"] = pd.read_csv(path).values
    
    return data


def ablation_A(data) -> list[dict]:
    """A: Raw features -> classical classifiers."""
    print("\n--- Ablation A: Raw Features -> Classical ---")
    
    results = []
    baselines_path = os.path.join(RESULTS_DIR, "classical_baselines.csv")
    if os.path.exists(baselines_path):
        df = pd.read_csv(baselines_path)
        for _, row in df.iterrows():
            r = row.to_dict()
            r["ablation"] = "A"
            r["pipeline"] = "Raw->Classical"
            results.append(r)
        print(f"  Loaded {len(results)} baseline results")
    return results


def ablation_B(data) -> list[dict]:
    """B: Raw features -> neural network (from baselines)."""
    print("\n--- Ablation B: Raw Features -> MLP ---")
    
    results = []
    baselines_path = os.path.join(RESULTS_DIR, "classical_baselines.csv")
    if os.path.exists(baselines_path):
        df = pd.read_csv(baselines_path)
        mlp_rows = df[df["model"].str.contains("MLP")]
        for _, row in mlp_rows.iterrows():
            r = row.to_dict()
            r["ablation"] = "B"
            r["pipeline"] = "Raw->MLP"
            results.append(r)
        print(f"  Loaded {len(results)} MLP results")
    return results


def ablation_C(data) -> list[dict]:
    """C: Raw -> encoder -> 16 latent features -> classical classifier."""
    print("\n--- Ablation C: Latent 16 -> Classical ---")
    
    results = []
    latent_path = os.path.join(RESULTS_DIR, "latent_validation.csv")
    if os.path.exists(latent_path):
        df = pd.read_csv(latent_path)
        for _, row in df.iterrows():
            r = row.to_dict()
            r["ablation"] = "C"
            r["pipeline"] = "Raw->Encoder->16dim->Classical"
            results.append(r)
        print(f"  Loaded {len(results)} latent classifier results")
    return results


def ablation_D(data) -> list[dict]:
    """D: Raw -> encoder -> 16 latent -> classical feature selection -> classifier."""
    print("\n--- Ablation D: Latent -> Classical FS -> Classifier ---")
    
    results = []
    
    # Load classical feature selection indices
    cfs_path = os.path.join(RESULTS_DIR, "selected_features_classical.json")
    if not os.path.exists(cfs_path):
        print("  Skipping: no classical feature selection results")
        return results
    
    with open(cfs_path) as f:
        cfs = json.load(f)
    
    # Use f_classif selected indices
    indices = cfs["f_classif"]["indices"]
    
    X_train_latent = data.get("X_latent_train")
    X_test_latent = data.get("X_latent_test")
    
    if X_train_latent is None:
        print("  Skipping: no latent features available")
        return results
    
    X_train_sel = X_train_latent[:, indices]
    X_test_sel = X_test_latent[:, indices]
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    for name, model in [
        ("ClassicalFS-LogReg", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
        ("ClassicalFS-SVM", SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            random_state=RANDOM_SEED)),
    ]:
        t0 = time.time()
        model.fit(X_train_sel, y_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        y_pred = model.predict(X_test_sel)
        inf_time = time.time() - t0
        
        y_prob = model.predict_proba(X_test_sel)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = compute_metrics(y_test, y_pred, y_prob, name)
        metrics["training_time"] = round(train_time, 4)
        metrics["inference_time"] = round(inf_time, 6)
        metrics["feature_space"] = f"classical_selected_{len(indices)}"
        metrics["num_features"] = len(indices)
        metrics["num_qubits"] = 0
        metrics["ablation"] = "D"
        metrics["pipeline"] = "Raw->Encoder->16dim->ClassicalFS->Classifier"
        
        print_metrics(metrics)
        results.append(metrics)
    
    return results


def ablation_E(data) -> list[dict]:
    """E: Raw -> encoder -> 16 latent -> quantum feature selection -> VQC."""
    print("\n--- Ablation E: Quantum Pipeline ---")
    
    results = []
    vqc_path = os.path.join(RESULTS_DIR, "vqc_test_metrics.json")
    if os.path.exists(vqc_path):
        with open(vqc_path) as f:
            metrics = json.load(f)
        metrics["ablation"] = "E"
        metrics["pipeline"] = "Raw->Encoder->16dim->QuantumFS->VQC"
        results.append(metrics)
        print(f"  Loaded VQC test metrics")
        print_metrics(metrics)
    else:
        print("  Skipping: VQC not yet trained")
    
    return results


def run_ablation_study() -> pd.DataFrame:
    """Run the complete ablation study."""
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)
    
    data = load_all_data()
    
    all_results = []
    all_results.extend(ablation_A(data))
    all_results.extend(ablation_B(data))
    all_results.extend(ablation_C(data))
    all_results.extend(ablation_D(data))
    all_results.extend(ablation_E(data))
    
    if not all_results:
        print("[ABLATION] No results available yet")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Save final comparison
    cols = [
        "model", "ablation", "pipeline", "feature_space", "num_features",
        "num_qubits", "accuracy", "precision", "recall", "specificity",
        "f1", "roc_auc", "pr_auc", "fn", "training_time", "inference_time"
    ]
    available_cols = [c for c in cols if c in df.columns]
    df_final = df[available_cols]
    
    final_path = os.path.join(RESULTS_DIR, "final_comparison.csv")
    df_final.to_csv(final_path, index=False)
    print(f"\n[ABLATION] Final comparison saved to {final_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    summary_cols = ["model", "ablation", "accuracy", "f1", "roc_auc", "fn"]
    summary_cols = [c for c in summary_cols if c in df_final.columns]
    print(df_final[summary_cols].to_string(index=False))
    
    # Save experiment log
    experiment_log = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_models": len(all_results),
        "ablations": list(df["ablation"].unique()) if "ablation" in df.columns else [],
        "random_seed": RANDOM_SEED,
    }
    log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    return df_final


if __name__ == "__main__":
    run_ablation_study()
