"""
Classical baseline classifier engine and feature benchmarking for Part 3.

Evaluates reference classical classifiers on:
1. Full feature space (30 raw scaled features)
2. Compact feature spaces (PCA & Autoencoder representations across 4, 8, 12, 16 dimensions)

Tracks class imbalance, reproducibility, model saving, and experiment registry.
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, MODELS_DIR, RESULTS_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT
)
from src.evaluation.metrics import compute_metrics, print_metrics, save_metrics


def get_class_imbalance_info(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Compute and format class distribution statistics across splits."""
    def get_counts(y):
        c0 = int(np.sum(y == 0))
        c1 = int(np.sum(y == 1))
        tot = len(y)
        return {"malignant_0": c0, "benign_1": c1, "total": tot, "pct_malignant": round(c0/tot*100, 2)}

    train_c = get_counts(y_train)
    val_c = get_counts(y_val)
    test_c = get_counts(y_test)
    
    imb_ratio = round(train_c["benign_1"] / train_c["malignant_0"], 3) if train_c["malignant_0"] > 0 else 1.0

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(k): round(float(v), 4) for k, v in zip(classes, weights)}

    return {
        "imbalance_ratio": imb_ratio,
        "class_weights": class_weights,
        "train_distribution": train_c,
        "val_distribution": val_c,
        "test_distribution": test_c,
    }


def train_eval_classifier(
    model,
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_space: str = "raw",
    n_features: int = 30
) -> Tuple[Dict[str, Any], Any]:
    """Train a classifier, evaluate on specified evaluation set, and return metrics."""
    # Training
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # Prediction
    t0 = time.time()
    y_pred = model.predict(X_eval)
    inference_time = time.time() - t0

    # Probabilities
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_eval)

    metrics = compute_metrics(y_eval, y_pred, y_prob, model_name)
    metrics["feature_space"] = feature_space
    metrics["num_features"] = n_features
    metrics["num_qubits"] = 0
    metrics["training_time"] = round(train_time, 4)
    metrics["inference_time"] = round(inference_time, 6)
    metrics["random_seed"] = RANDOM_SEED

    return metrics, model


def run_classical_baselines(save_models: bool = True) -> List[Dict[str, Any]]:
    """
    Run baseline reference classifiers on FULL preprocessed 30-feature data.
    """
    print("\n" + "=" * 60)
    print("OBJECTIVE A: CLASSICAL BASELINE CLASSIFIERS (FULL 30 FEATURES)")
    print("=" * 60)

    # Load preprocessed splits
    X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv")).squeeze().values
    y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv")).squeeze().values
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv")).squeeze().values

    imb_info = get_class_imbalance_info(y_train, y_val, y_test)
    print(f"[BASELINE] Class Imbalance Ratio: {imb_info['imbalance_ratio']} : 1")
    print(f"[BASELINE] Balanced Class Weights: {imb_info['class_weights']}")

    models_dict = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs"
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Classical MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_SEED,
            early_stopping=True, validation_fraction=0.15, learning_rate="adaptive"
        ),
    }

    baseline_dir = os.path.join(MODELS_DIR, "baseline")
    exp_dir = os.path.join(PROJECT_ROOT, "experiments", "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    all_results = []
    
    for name, model in models_dict.items():
        print(f"\n[BASELINE] Training {name} on {X_train.shape[1]} features...")
        
        # 1. Validation metrics
        val_metrics, _ = train_eval_classifier(
            model, f"{name} (Val)", X_train.values, y_train, X_val.values, y_val,
            feature_space="raw_30", n_features=X_train.shape[1]
        )

        # 2. Test metrics (isolated test set)
        test_metrics, trained_model = train_eval_classifier(
            model, name, X_train.values, y_train, X_test.values, y_test,
            feature_space="raw_30", n_features=X_train.shape[1]
        )
        test_metrics["val_accuracy"] = val_metrics["accuracy"]
        test_metrics["val_f1"] = val_metrics["f1"]

        print_metrics(test_metrics)
        all_results.append(test_metrics)

        if save_models:
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = os.path.join(baseline_dir, f"model_{safe_name}.joblib")
            joblib.dump(trained_model, save_path)
            
            # Save configuration
            cfg = {
                "model_name": name,
                "hyperparameters": str(trained_model.get_params()),
                "feature_count": X_train.shape[1],
                "random_seed": RANDOM_SEED,
                "class_weights": imb_info["class_weights"],
            }
            with open(os.path.join(baseline_dir, f"config_{safe_name}.json"), "w") as f:
                json.dump(cfg, f, indent=2)

    # Save metrics to results and experiments
    save_metrics(all_results, "classical_baselines.csv")
    
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({
            "imbalance_info": imb_info,
            "results": all_results
        }, f, indent=2)

    return all_results


def run_compact_feature_experiments(
    target_dims: list[int] = [4, 8, 12, 16]
) -> List[Dict[str, Any]]:
    """
    Evaluate baseline classifiers (Logistic Regression & SVM) on compact representation spaces (PCA & Autoencoder).
    Compares FULL 30-feature baseline vs compact 4D, 8D, 12D, 16D representations.
    """
    print("\n" + "=" * 60)
    print("OBJECTIVE B: COMPACT FEATURE CLASSIFIER EVALUATION & COMPARISON")
    print("=" * 60)

    features_dir = os.path.join(PROJECT_ROOT, "features", "classical")
    y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv")).squeeze().values
    y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv")).squeeze().values
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv")).squeeze().values

    compact_results = []
    registry = []

    methods = ["pca", "autoencoder"]

    for method in methods:
        for dim in target_dims:
            tr_file = os.path.join(features_dir, f"X_train_{method}_{dim}.csv")
            va_file = os.path.join(features_dir, f"X_val_{method}_{dim}.csv")
            te_file = os.path.join(features_dir, f"X_test_{method}_{dim}.csv")

            if not (os.path.exists(tr_file) and os.path.exists(va_file) and os.path.exists(te_file)):
                print(f"Skipping {method} {dim}D: Feature files not found.")
                continue

            X_tr = pd.read_csv(tr_file).values
            X_va = pd.read_csv(va_file).values
            X_te = pd.read_csv(te_file).values

            classifiers = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
                ),
                "SVM (RBF)": SVC(
                    kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED
                ),
            }

            for clf_name, clf in classifiers.items():
                model_identifier = f"{clf_name} [{method.upper()}-{dim}D]"
                
                # Val evaluation
                val_m, _ = train_eval_classifier(
                    clf, f"{model_identifier} (Val)", X_tr, y_train, X_va, y_val,
                    feature_space=f"{method}_{dim}", n_features=dim
                )

                # Test evaluation
                test_m, trained_clf = train_eval_classifier(
                    clf, model_identifier, X_tr, y_train, X_te, y_test,
                    feature_space=f"{method}_{dim}", n_features=dim
                )
                test_m["val_accuracy"] = val_m["accuracy"]
                test_m["val_f1"] = val_m["f1"]
                test_m["method"] = method
                test_m["compression_ratio"] = round(30 / dim, 2)

                compact_results.append(test_m)
                
                exp_entry = {
                    "id": f"experiment_{len(registry)+1:03d}",
                    "date_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": clf_name,
                    "reduction_method": method,
                    "input_features": 30,
                    "output_features": dim,
                    "accuracy": test_m["accuracy"],
                    "precision": test_m["precision"],
                    "recall": test_m["recall"],
                    "specificity": test_m["specificity"],
                    "f1": test_m["f1"],
                    "roc_auc": test_m["roc_auc"],
                    "pr_auc": test_m["pr_auc"],
                    "fn": test_m["fn"],
                    "training_time": test_m["training_time"],
                    "inference_time": test_m["inference_time"],
                    "random_seed": RANDOM_SEED,
                }
                registry.append(exp_entry)

                print(f"  {model_identifier:30s} | Test Acc: {test_m['accuracy']:.4f} | Recall: {test_m['recall']:.4f} | F1: {test_m['f1']:.4f} | ROC-AUC: {test_m['roc_auc']:.4f}")

    # Save compact results & registry
    save_metrics(compact_results, "compact_feature_baselines.csv")

    exp_reg_path = os.path.join(PROJECT_ROOT, "experiments", "experiment_registry.json")
    with open(exp_reg_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n[EXPERIMENTS] Compact feature benchmarking complete. Registry saved to {exp_reg_path}")
    return compact_results


if __name__ == "__main__":
    run_classical_baselines()
    run_compact_feature_experiments()
