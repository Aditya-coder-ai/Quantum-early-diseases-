"""
Classical baseline models for the medical disease detection task.

Implements:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. Classical Multi-Layer Perceptron (MLP)

All models use the same train/val/test splits.
Evaluation uses the test set ONLY for final metrics.
Validation set is used for model selection where applicable.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import RANDOM_SEED, MODELS_DIR, RESULTS_DIR, DATA_PROCESSED_DIR
from src.evaluation.metrics import compute_metrics, print_metrics, save_metrics


def load_processed_data() -> dict:
    """Load preprocessed train/val/test splits."""
    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, f"X_{split}.csv"))
        data[f"y_{split}"] = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")).squeeze()
    return data


def get_class_weights(y_train: np.ndarray) -> dict:
    """Compute balanced class weights from training data."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def train_and_evaluate_model(
    model, model_name: str, X_train, y_train, X_test, y_test
) -> dict:
    """Train a model, evaluate on test set, return metrics and timing."""
    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    
    # Probabilities if available
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    
    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob, model_name)
    metrics["training_time"] = round(train_time, 4)
    metrics["inference_time"] = round(inference_time, 6)
    metrics["feature_space"] = "raw"
    metrics["num_features"] = X_train.shape[1]
    metrics["num_qubits"] = 0
    
    return metrics, model


def run_classical_baselines() -> list[dict]:
    """Run all classical baseline models and return metrics."""
    print("\n" + "="*60)
    print("CLASSICAL BASELINE MODELS")
    print("="*60)
    
    data = load_processed_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    class_weights = get_class_weights(y_train.values)
    print(f"[BASELINES] Class weights: {class_weights}")
    print(f"[BASELINES] Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Classical MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate="adaptive",
        ),
    }
    
    all_metrics = []
    saved_models = {}
    
    for name, model in models.items():
        print(f"\n[BASELINES] Training: {name}")
        metrics, trained_model = train_and_evaluate_model(
            model, name, X_train, y_train, X_test, y_test
        )
        print_metrics(metrics)
        all_metrics.append(metrics)
        saved_models[name] = trained_model
        
        # Save model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(MODELS_DIR, f"baseline_{safe_name}.joblib")
        joblib.dump(trained_model, model_path)
    
    # Save results
    save_metrics(all_metrics, "classical_baselines.csv")
    
    return all_metrics


# Alias for pipeline runner
run_all_baselines = run_classical_baselines


if __name__ == "__main__":
    results = run_classical_baselines()
    
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    summary_cols = ["model", "accuracy", "precision", "recall",
                    "specificity", "f1", "roc_auc", "fn"]
    print(df[summary_cols].to_string(index=False))
