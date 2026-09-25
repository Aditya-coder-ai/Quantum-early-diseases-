"""
Visualization module for generating all research plots and figures.

Generates:
- Confusion matrices (Hybrid VQC and baselines)
- ROC curves (multi-model comparison)
- Precision-Recall curves
- Model metric comparison bar charts
- Autoencoder and VQC training/validation loss curves
- Latent-space t-SNE visualization
- Quantum feature selection importance charts
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import RESULTS_DIR, DATA_PROCESSED_DIR, RANDOM_SEED


def setup_style():
    """Set consistent publication plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot a confusion matrix heatmap."""
    setup_style()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"]
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_model_comparison(results_df, metric_cols=None, save_path=None):
    """Bar chart comparing models across key metrics."""
    setup_style()
    if metric_cols is None:
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    available_cols = [c for c in metric_cols if c in results_df.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df))
    width = 0.15

    for i, col in enumerate(available_cols):
        vals = results_df[col].astype(float)
        bars = ax.bar(x + i * width, vals, width, label=col.upper())
        for bar, val in zip(bars, vals):
            if not pd.isna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7
                )

    ax.set_xlabel("Model / Pipeline Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Medical Diagnosis System: Model Comparison")
    ax.set_xticks(x + width * (len(available_cols) - 1) / 2)
    ax.set_xticklabels(results_df["model"], rotation=30, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_training_loss(history, title="Training Loss", save_path=None):
    """Plot training and validation loss curves."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2, color="#2980b9")
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, color="#e74c3c")

    if "best_epoch" in history:
        best_ep = history["best_epoch"] + 1
        ax.axvline(
            x=best_ep, color="black", linestyle="--", alpha=0.7,
            label=f"Best Checkpoint (Epoch {best_ep})"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_latent_tsne(latent_features, labels, title="Autoencoder Latent Space (16-D via t-SNE)", save_path=None, seed=RANDOM_SEED):
    """t-SNE visualization of the 16D latent space."""
    setup_style()
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    embedding = tsne.fit_transform(latent_features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels,
        cmap="coolwarm", alpha=0.8, s=45, edgecolors="white", linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Malignant (0)", "Benign (1)"])
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_feature_importance(importance_scores, selected_indices, title="Quantum Feature Selection (8 of 16)", save_path=None):
    """Plot feature importance with selected features highlighted."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(importance_scores)
    x = np.arange(n)

    colors = ["#2ecc71" if i in selected_indices else "#bdc3c7" for i in range(n)]
    ax.bar(x, importance_scores, color=colors, edgecolor="white", width=0.6)

    ax.set_xlabel("Latent Feature Dimension")
    ax.set_ylabel("Relevance Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Latent_{i}" for i in range(n)], rotation=45)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Selected by Quantum QFS (8 Qubits)"),
        Patch(facecolor="#bdc3c7", label="Pruned / Redundant Feature"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_roc_pr_curves(save_dir=None):
    """Generate ROC and Precision-Recall comparison curves."""
    setup_style()
    if save_dir is None:
        save_dir = os.path.join(RESULTS_DIR, "plots")

    # Load test labels
    y_test_path = os.path.join(DATA_PROCESSED_DIR, "y_test.csv")
    if not os.path.exists(y_test_path):
        return
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Load VQC predictions
    from src.quantum.vqc import HybridVQCModule, AngleScaler
    import torch
    
    models_dict = {}
    
    # 1. Hybrid VQC
    vqc_path = os.path.join(RESULTS_DIR, "..", "models", "vqc_model.pt")
    sel_test_path = os.path.join(RESULTS_DIR, "selected_features_test.csv")
    sel_train_path = os.path.join(RESULTS_DIR, "selected_features_train.csv")
    if os.path.exists(vqc_path) and os.path.exists(sel_test_path) and os.path.exists(sel_train_path):
        X_train_sel = pd.read_csv(sel_train_path).values
        X_test_sel = pd.read_csv(sel_test_path).values
        scaler = AngleScaler().fit(X_train_sel)
        X_test_ang = scaler.transform(X_test_sel)
        
        vqc = HybridVQCModule()
        vqc.load_state_dict(torch.load(vqc_path, weights_only=True))
        models_dict["Hybrid VQC"] = vqc.predict_proba(X_test_ang)

    # 2. Classical Baselines
    import joblib
    for name, filename in [
        ("Logistic Regression", "baseline_logistic_regression.joblib"),
        ("SVM (RBF)", "baseline_svm_rbf.joblib"),
        ("Random Forest", "baseline_random_forest.joblib"),
    ]:
        model_file = os.path.join(RESULTS_DIR, "..", "models", filename)
        X_test_raw = os.path.join(DATA_PROCESSED_DIR, "X_test.csv")
        if os.path.exists(model_file) and os.path.exists(X_test_raw):
            clf = joblib.load(model_file)
            X_test_mat = pd.read_csv(X_test_raw).values
            if hasattr(clf, "predict_proba"):
                models_dict[name] = clf.predict_proba(X_test_mat)[:, 1]

    if not models_dict:
        return

    # ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("ROC Curves Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), bbox_inches="tight")
    plt.close()

    # Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_score = auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_score:.4f})", linewidth=2)

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves Comparison")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curves.png"), bbox_inches="tight")
    plt.close()

    # Confusion matrix for Hybrid VQC
    if "Hybrid VQC" in models_dict:
        vqc_preds = (models_dict["Hybrid VQC"] >= 0.5).astype(int)
        plot_confusion_matrix(
            y_test, vqc_preds, "Hybrid VQC",
            save_path=os.path.join(save_dir, "confusion_matrix_vqc.png")
        )


def generate_all_plots():
    """Generate all available plots from saved results."""
    print("\n" + "=" * 60)
    print("GENERATING RESEARCH PLOTS & FIGURES")
    print("=" * 60)

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Model comparison
    try:
        final_df = pd.read_csv(os.path.join(RESULTS_DIR, "final_comparison.csv"))
        plot_model_comparison(
            final_df,
            save_path=os.path.join(plots_dir, "model_comparison.png")
        )
        print("[PLOTS] Model comparison bar chart saved")
    except Exception as e:
        print(f"[PLOTS] Skipping model comparison: {e}")

    # 2. Autoencoder training loss
    try:
        with open(os.path.join(RESULTS_DIR, "autoencoder_history.json")) as f:
            ae_hist = json.load(f)
        plot_training_loss(
            ae_hist, title="Autoencoder Training & Validation Loss",
            save_path=os.path.join(plots_dir, "autoencoder_loss.png")
        )
        print("[PLOTS] Autoencoder loss curve saved")
    except Exception as e:
        print(f"[PLOTS] Skipping AE loss: {e}")

    # 3. VQC training loss
    try:
        with open(os.path.join(RESULTS_DIR, "vqc_training_history.json")) as f:
            vqc_hist = json.load(f)
        plot_training_loss(
            vqc_hist, title="Hybrid VQC Training & Validation Loss",
            save_path=os.path.join(plots_dir, "vqc_loss.png")
        )
        print("[PLOTS] VQC loss curve saved")
    except Exception as e:
        print(f"[PLOTS] Skipping VQC loss: {e}")

    # 4. Latent space t-SNE
    try:
        latent_train = pd.read_csv(
            os.path.join(RESULTS_DIR, "latent_features_train.csv")
        ).values
        y_train = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, "y_train.csv")
        ).values.ravel()
        plot_latent_tsne(
            latent_train, y_train,
            save_path=os.path.join(plots_dir, "latent_tsne.png")
        )
        print("[PLOTS] Latent space t-SNE visualization saved")
    except Exception as e:
        print(f"[PLOTS] Skipping t-SNE: {e}")

    # 5. Feature importance / selection
    try:
        with open(os.path.join(RESULTS_DIR, "selected_features_quantum.json")) as f:
            qfs = json.load(f)
        plot_feature_importance(
            qfs["feature_importance"], qfs["selected_indices"],
            title="Quantum Feature Selection (8 Qubits Selected out of 16 Latent Dimensions)",
            save_path=os.path.join(plots_dir, "feature_selection.png")
        )
        print("[PLOTS] Feature selection plot saved")
    except Exception as e:
        print(f"[PLOTS] Skipping feature selection: {e}")

    # 6. ROC, PR, and Confusion Matrix
    try:
        plot_roc_pr_curves(save_dir=plots_dir)
        print("[PLOTS] ROC curves, PR curves, and VQC Confusion Matrix saved")
    except Exception as e:
        print(f"[PLOTS] Skipping ROC/PR curves: {e}")

    print(f"\n[PLOTS] All figures successfully written to {plots_dir}")


if __name__ == "__main__":
    generate_all_plots()
