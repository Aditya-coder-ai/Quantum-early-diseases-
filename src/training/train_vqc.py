"""
Hybrid Classical-Quantum VQC training pipeline.

Pipeline:
    Selected latent features (8-dim)
    -> Angle encoding (RY rotation angles in [0, pi])
    -> 8-qubit Variational Quantum Circuit (2 layers, circular entanglement)
    -> PauliZ measurement
    -> Classical linear calibration head
    -> Binary cross-entropy with logits (class-weighted for imbalance)
    -> Adam optimization with adjoint differentiation
    -> Early stopping on validation loss

Strictly prevents data leakage: AngleScaler and class weights fit on train set only.
Validation set used solely for early stopping and model selection.
Test set reserved strictly for final reporting.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, RESULTS_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
    NUM_QUBITS, VQC_NUM_LAYERS, VQC_LEARNING_RATE, VQC_EPOCHS,
    VQC_BATCH_SIZE, VQC_PATIENCE, SELECTED_DIM
)
from src.quantum.vqc import HybridVQCModule, AngleScaler, vqc_predict_batch
from src.evaluation.metrics import compute_metrics, print_metrics


def load_selected_features() -> dict:
    """Load selected 8-dim features and labels."""
    data = {}
    for split in ["train", "val", "test"]:
        X_path = os.path.join(RESULTS_DIR, f"selected_features_{split}.csv")
        y_path = os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")
        data[f"X_{split}"] = pd.read_csv(X_path).values.astype(np.float32)
        data[f"y_{split}"] = pd.read_csv(y_path).values.ravel().astype(np.float32)
    return data


def train_vqc(
    epochs: int = 60,
    lr: float = 0.02,
    batch_size: int = 32,
    patience: int = 15,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Train the Hybrid VQC model using PyTorch and adjoint differentiation.

    Returns:
        Dict with trained model, history, and test metrics.
    """
    print("\n" + "=" * 60)
    print("HYBRID CLASSICAL-QUANTUM VQC TRAINING")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load data
    data = load_selected_features()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    assert X_train.shape[1] == SELECTED_DIM, (
        f"Expected {SELECTED_DIM} features, got {X_train.shape[1]}"
    )
    print(f"[VQC-TRAIN] Dataset splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Angle encoding (fit on training data only)
    angle_scaler = AngleScaler()
    X_train_angles = angle_scaler.fit_transform(X_train)
    X_val_angles = angle_scaler.transform(X_val)
    X_test_angles = angle_scaler.transform(X_test)

    # Assert valid angle ranges
    assert not np.isnan(X_train_angles).any(), "NaN detected in training angles"
    assert not np.isnan(X_val_angles).any(), "NaN detected in validation angles"
    assert not np.isnan(X_test_angles).any(), "NaN detected in test angles"
    assert X_train_angles.min() >= 0.0, "Negative angles detected"
    assert X_train_angles.max() <= np.pi + 1e-5, "Angles exceed pi"
    print(f"[VQC-TRAIN] Angle encoding verified. Min={X_train_angles.min():.4f}, Max={X_train_angles.max():.4f}")

    # 3. Handle class imbalance in loss function (computed from train only)
    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[VQC-TRAIN] Class imbalance ratio (Train): Neg={int(n_neg)}, Pos={int(n_pos)} (weight={pos_weight.item():.3f})")

    # 4. Prepare PyTorch tensors and DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train_angles, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    X_val_t = torch.tensor(X_val_angles, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_angles, dtype=torch.float32)

    # 5. Initialize Model & Optimizer
    model = HybridVQCModule(n_qubits=NUM_QUBITS, n_layers=VQC_NUM_LAYERS, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }

    best_state = None
    patience_counter = 0
    t_start = time.time()

    print(f"[VQC-TRAIN] Starting optimization: {epochs} epochs max, patience={patience}, batch_size={batch_size}")

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = float(np.mean(batch_losses))

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = float(np.mean(val_preds == y_val))

            # Train accuracy
            train_probs = model.predict_proba(X_train_angles)
            train_preds = (train_probs >= 0.5).astype(int)
            train_acc = float(np.mean(train_preds == y_train))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Check early stopping
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter >= patience:
            print(
                f"  Epoch {epoch+1:2d}/{epochs}: "
                f"Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f} | "
                f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch+1} (Best epoch: {history['best_epoch']+1})")
            break

    total_training_time = round(time.time() - t_start, 2)
    history["training_time"] = total_training_time

    # Load best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n[VQC-TRAIN] Training complete in {total_training_time}s. Best Val Loss={history['best_val_loss']:.4f}")

    # 6. Final Test Evaluation (Test split never touched until now)
    print("\n--- Final Evaluation on Test Set ---")
    t_inf_start = time.time()
    test_probs = model.predict_proba(X_test_angles)
    test_inf_time = round(time.time() - t_inf_start, 4)
    test_preds = (test_probs >= 0.5).astype(int)

    test_metrics = compute_metrics(y_test, test_preds, test_probs, "Hybrid VQC")
    test_metrics["training_time"] = total_training_time
    test_metrics["inference_time"] = test_inf_time
    test_metrics["feature_space"] = "quantum_selected_8"
    test_metrics["num_features"] = SELECTED_DIM
    test_metrics["num_qubits"] = NUM_QUBITS
    test_metrics["pipeline"] = "Raw->Encoder->16dim->QuantumFS->VQC"
    test_metrics["ablation"] = "E"
    print_metrics(test_metrics)

    # 7. Save artifacts
    save_artifacts(model, angle_scaler, history, test_metrics)

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "angle_scaler": angle_scaler
    }


def save_artifacts(model, angle_scaler, history, test_metrics):
    """Save all model checkpoints and result files."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save PyTorch state dict
    torch_model_path = os.path.join(MODELS_DIR, "vqc_model.pt")
    torch.save(model.state_dict(), torch_model_path)

    # Save pure weights numpy array for hardware portability
    weights_path = os.path.join(MODELS_DIR, "vqc_weights.npy")
    np.save(weights_path, model.weights.detach().cpu().numpy())

    # Save AngleScaler
    scaler_path = os.path.join(MODELS_DIR, "angle_scaler.json")
    with open(scaler_path, "w") as f:
        json.dump({
            "min_vals": angle_scaler.min_vals.tolist(),
            "range_vals": angle_scaler.range_vals.tolist(),
        }, f, indent=2)

    # Save history
    hist_path = os.path.join(RESULTS_DIR, "vqc_training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save test metrics
    metrics_path = os.path.join(RESULTS_DIR, "vqc_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"[VQC-SAVE] Model saved to {torch_model_path}")
    print(f"[VQC-SAVE] History saved to {hist_path}")
    print(f"[VQC-SAVE] Test metrics saved to {metrics_path}")


if __name__ == "__main__":
    train_vqc()
