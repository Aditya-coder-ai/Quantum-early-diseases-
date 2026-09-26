"""
Latent space validation: train classical classifiers on the 16-dim
latent representation to verify it preserves discriminative information.

This is a critical ablation step — if the latent features don't work
for classical classifiers, they won't work for quantum circuits either.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import RANDOM_SEED, RESULTS_DIR, DATA_PROCESSED_DIR
from src.evaluation.metrics import compute_metrics, print_metrics, save_metrics


def load_latent_data() -> dict:
    """Load latent features and labels."""
    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = pd.read_csv(
            os.path.join(RESULTS_DIR, f"latent_features_{split}.csv"))
        data[f"y_{split}"] = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")).squeeze()
    return data


def compute_latent_clustering_metrics(X_latent: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate how well benign vs malignant classes separate in 16D latent space."""
    return {
        "silhouette_score": round(float(silhouette_score(X_latent, y)), 4),
        "calinski_harabasz_score": round(float(calinski_harabasz_score(X_latent, y)), 4),
        "davies_bouldin_score": round(float(davies_bouldin_score(X_latent, y)), 4),
    }


def validate_latent_classifiers():
    """Train and evaluate classifiers on the latent representation."""
    print("\n" + "="*60)
    print("LATENT SPACE VALIDATION")
    print("="*60)
    
    data = load_latent_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    print(f"[LATENT-VAL] Latent train shape: {X_train.shape}")
    print(f"[LATENT-VAL] Latent test shape:  {X_test.shape}")

    # Evaluate unsupervised clustering metrics on test set
    cluster_metrics = compute_latent_clustering_metrics(X_test.values, y_test.values)
    print(f"[LATENT-VAL] Latent Clustering Metrics (Test): "
          f"Silhouette={cluster_metrics['silhouette_score']} | "
          f"Calinski-Harabasz={cluster_metrics['calinski_harabasz_score']} | "
          f"Davies-Bouldin={cluster_metrics['davies_bouldin_score']}")
    
    with open(os.path.join(RESULTS_DIR, "latent_clustering_metrics.json"), "w") as f:
        json.dump(cluster_metrics, f, indent=2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        models = {
            "Latent-LogReg": LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=RANDOM_SEED
            ),
            "Latent-SVM": SVC(
                kernel="rbf", class_weight="balanced",
                probability=True, random_state=RANDOM_SEED
            ),
            "Latent-RandomForest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1
            ),
        }
    
    all_metrics = []
    for name, model in models.items():
        print(f"\n[LATENT-VAL] Training: {name}")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = compute_metrics(y_test, y_pred, y_prob, name)
        metrics["feature_space"] = "latent_16"
        metrics["num_features"] = X_train.shape[1]
        metrics["num_qubits"] = 0
        print_metrics(metrics)
        all_metrics.append(metrics)
    
    save_metrics(all_metrics, "latent_validation.csv")
    
    # Compare with raw-feature baselines
    print(f"\n{'='*60}")
    print("COMPARISON: Raw vs Latent Features")
    print(f"{'='*60}")
    
    try:
        raw_df = pd.read_csv(os.path.join(RESULTS_DIR, "classical_baselines.csv"))
        latent_df = pd.DataFrame(all_metrics)
        
        for _, raw_row in raw_df.iterrows():
            print(f"  {raw_row['model']:25s} (raw-30):   "
                  f"Acc={raw_row['accuracy']:.4f}  F1={raw_row['f1']:.4f}  "
                  f"AUC={raw_row['roc_auc']:.4f}")
        for _, lat_row in latent_df.iterrows():
            print(f"  {lat_row['model']:25s} (latent-16):"
                  f" Acc={lat_row['accuracy']:.4f}  F1={lat_row['f1']:.4f}  "
                  f"AUC={lat_row['roc_auc']:.4f}")
    except FileNotFoundError:
        print("  (Raw baseline results not yet available for comparison)")
    
    return all_metrics


# Alias for pipeline runner
run_latent_validation = validate_latent_classifiers


if __name__ == "__main__":
    validate_latent_classifiers()
