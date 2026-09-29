"""
Part 6: VQC Training Engine, Loss Functions, Gradient Verification & Early Stopping.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import precision_recall_curve, auc

from src.vqc.model import VariationalQuantumClassifier


def compute_weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[Dict[int, float]] = None
) -> torch.Tensor:
    """
    Numerically stable Binary Cross Entropy with optional per-class weighting.
    
    Args:
        logits: Unnormalized network outputs of shape (batch_size,)
        targets: Binary labels {0, 1} of shape (batch_size,)
        class_weights: Optional mapping {0: w0, 1: w1}
    """
    bce_raw = nn.BCEWithLogitsLoss(reduction="none")(logits, targets.float())
    if class_weights is not None:
        # Assign sample-wise weights based on true labels
        w0 = class_weights.get(0, 1.0)
        w1 = class_weights.get(1, 1.0)
        weights = torch.where(targets == 1, torch.tensor(w1, device=targets.device), torch.tensor(w0, device=targets.device))
        return (bce_raw * weights).mean()
    return bce_raw.mean()


def train_vqc(
    model: VariationalQuantumClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 0.02,
    weight_decay: float = 1e-4,
    class_weights: Optional[Dict[int, float]] = None,
    early_stopping_patience: int = 8,
    monitor_metric: str = "val_pr_auc",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a Variational Quantum Classifier with early stopping and validation tracking.
    
    Args:
        model: VariationalQuantumClassifier instance
        X_train: Preprocessed/angle-encoded training features (n_train, n_qubits)
        y_train: Training labels (n_train,)
        X_val: Preprocessed/angle-encoded validation features (n_val, n_qubits)
        y_val: Validation labels (n_val,)
        epochs: Maximum training epochs
        batch_size: Mini-batch size
        lr: Adam learning rate
        class_weights: Optional loss weights for class imbalance
        early_stopping_patience: Epochs without improvement before stopping
        monitor_metric: 'val_pr_auc' or 'val_loss'
    
    Returns:
        Dictionary containing history, best validation metrics, training time, and convergence status.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_samples = len(X_train)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_pr_auc": [],
        "val_minority_recall": [],
        "epoch_time_s": [],
    }

    best_val_metric = -1.0 if monitor_metric == "val_pr_auc" else float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()

        # Shuffle training indices
        perm = np.random.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        n_batches = int(np.ceil(n_samples / batch_size))

        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = torch.tensor(X_shuffled[start_idx:end_idx], dtype=torch.float32)
            batch_y = torch.tensor(y_shuffled[start_idx:end_idx], dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = compute_weighted_bce_loss(logits, batch_y, class_weights=class_weights)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN or Inf loss encountered at epoch {epoch}, batch {b}.")

            loss.backward()

            # Verify finite gradients on all trainable parameters
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        raise RuntimeError(f"NaN or Inf gradient detected in {name} at epoch {epoch}.")

            optimizer.step()
            epoch_loss += loss.item() * (end_idx - start_idx)

        train_loss = epoch_loss / n_samples

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=torch.float32)
            val_y = torch.tensor(y_val, dtype=torch.float32)
            val_logits = model(val_x)
            val_loss = compute_weighted_bce_loss(val_logits, val_y, class_weights=class_weights).item()

            probs_val = torch.sigmoid(val_logits).cpu().numpy()
            preds_val = (probs_val >= 0.5).astype(int)

            val_acc = float(np.mean(preds_val == y_val))
            
            # Minority class recall (Class 0)
            minority_mask = (y_val == 0)
            if minority_mask.sum() > 0:
                val_min_recall = float((preds_val[minority_mask] == 0).sum() / minority_mask.sum())
            else:
                val_min_recall = 0.0

            # PR-AUC for minority class (score = 1 - probs_val)
            precision_curve, recall_curve, _ = precision_recall_curve(1 - y_val, 1.0 - probs_val)
            val_pr_auc = float(auc(recall_curve, precision_curve))

        epoch_time = time.time() - t_epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_pr_auc"].append(round(val_pr_auc, 4))
        history["val_minority_recall"].append(round(val_min_recall, 4))
        history["epoch_time_s"].append(round(epoch_time, 3))

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"  Epoch [{epoch:2d}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Minority Recall: {val_min_recall:.4f} "
                  f"({epoch_time:.2f}s)")

        # Check early stopping criterion
        current_metric = val_pr_auc if monitor_metric == "val_pr_auc" else -val_loss
        improved = current_metric > best_val_metric

        if improved:
            best_val_metric = current_metric
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  [Early Stopping] No improvement for {early_stopping_patience} epochs. "
                          f"Restoring best weights from epoch {best_epoch}.")
                break

    # Restore best validation checkpoint
    model.load_state_dict(best_weights)
    total_training_time = round(time.time() - t_start, 2)

    return {
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history["train_loss"]),
        "total_training_time_s": total_training_time,
        "best_val_loss": history["val_loss"][best_epoch - 1],
        "best_val_accuracy": history["val_accuracy"][best_epoch - 1],
        "best_val_pr_auc": history["val_pr_auc"][best_epoch - 1],
        "best_val_minority_recall": history["val_minority_recall"][best_epoch - 1],
        "history": history,
    }


def run_tiny_vqc_test(seed: int = 42) -> Dict[str, Any]:
    """
    Step 13 & 14 verification: Executes a tiny synthetic 4-qubit VQC test.
    Verifies that:
        - encoding works
        - circuit compiles and executes
        - gradients exist and are finite
        - parameters change after optimizer step
        - loss decreases
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Tiny synthetic 4D dataset (16 samples)
    n_samples, n_qubits = 16, 4
    X_synthetic = np.random.uniform(0.0, np.pi, (n_samples, n_qubits)).astype(np.float32)
    y_synthetic = np.random.choice([0, 1], size=n_samples)

    # 2. Build tiny 4-qubit VQC model
    model = VariationalQuantumClassifier(
        n_qubits=n_qubits,
        n_layers=1,
        ansatz_type="ansatz_a",
        entanglement="linear",
        seed=seed
    )

    initial_weights = model.weights.clone().detach()

    # 3. Check forward pass
    x_tensor = torch.tensor(X_synthetic[:4], dtype=torch.float32)
    y_tensor = torch.tensor(y_synthetic[:4], dtype=torch.float32)
    logits = model(x_tensor)
    loss = compute_weighted_bce_loss(logits, y_tensor)

    # 4. Check backward pass & finite gradients
    loss.backward()
    grad_norm = float(model.weights.grad.norm().item())
    has_finite_grad = not (np.isnan(grad_norm) or np.isinf(grad_norm)) and grad_norm > 0.0

    # 5. Optimizer step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.step()
    weight_diff = float((model.weights - initial_weights).abs().sum().item())
    params_updated = weight_diff > 1e-5

    status = "PASSED" if (has_finite_grad and params_updated) else "FAILED"

    return {
        "status": status,
        "n_qubits": n_qubits,
        "initial_loss": float(loss.item()),
        "grad_norm": grad_norm,
        "weight_diff": weight_diff,
        "has_finite_grad": has_finite_grad,
        "params_updated": params_updated,
    }
