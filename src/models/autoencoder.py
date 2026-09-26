"""
Classical Autoencoder for feature extraction.

Architecture:
    Encoder: Input(30) -> 64 -> 32 -> 16 (latent)
    Decoder: 16 -> 32 -> 64 -> Output(30)

Trained on TRAINING DATA ONLY with validation-based early stopping.
Produces a 16-dimensional latent representation for downstream quantum processing.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, MODELS_DIR, RESULTS_DIR, DATA_PROCESSED_DIR,
    LATENT_DIM, AE_HIDDEN_DIMS, AE_LEARNING_RATE, AE_BATCH_SIZE,
    AE_EPOCHS, AE_PATIENCE
)


def set_seed(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder with configurable architecture.
    
    Encoder: input_dim -> hidden[0] -> hidden[1] -> latent_dim
    Decoder: latent_dim -> hidden[1] -> hidden[0] -> input_dim
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, latent)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def load_processed_splits() -> dict:
    """Load preprocessed data splits as tensors."""
    data = {}
    for split in ["train", "val", "test"]:
        X = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, f"X_{split}.csv"))
        y = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")).squeeze()
        data[f"X_{split}"] = torch.tensor(X.values.copy(), dtype=torch.float32)
        data[f"y_{split}"] = torch.tensor(y.values.copy(), dtype=torch.long)
    return data


def train_autoencoder(
    model: Autoencoder,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    epochs: int = AE_EPOCHS,
    batch_size: int = AE_BATCH_SIZE,
    lr: float = AE_LEARNING_RATE,
    patience: int = AE_PATIENCE,
) -> dict:
    """
    Train the autoencoder with early stopping on validation loss.
    
    Returns training history dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True
    )
    
    history = {
        "train_loss": [], "val_loss": [],
        "best_epoch": 0, "best_val_loss": float("inf")
    }
    
    best_state = None
    patience_counter = 0
    
    print(f"[AE] Training on {device} | Epochs: {epochs} | "
          f"LR: {lr} | Batch: {batch_size}")
    
    t_start = time.time()
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            x_recon, _ = model(batch_x)
            loss = criterion(x_recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            X_val_dev = X_val.to(device)
            x_recon_val, _ = model(X_val_dev)
            val_loss = criterion(x_recon_val, X_val_dev).item()
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"Train={avg_train_loss:.6f} Val={val_loss:.6f} "
                  f"(best={history['best_val_loss']:.6f} @ {history['best_epoch']+1})")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(best @ {history['best_epoch']+1})")
            break
    
    history["training_time"] = round(time.time() - t_start, 2)
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    print(f"[AE] Training complete in {history['training_time']}s. "
          f"Best val loss: {history['best_val_loss']:.6f} "
          f"at epoch {history['best_epoch']+1}")
    
    return history


def extract_latent_features(
    model: Autoencoder, X: torch.Tensor
) -> np.ndarray:
    """Extract latent features using the trained encoder."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = model.encode(X.to(device))
    return z.cpu().numpy()


def evaluate_reconstruction(
    model: Autoencoder,
    data: dict[str, torch.Tensor]
) -> dict:
    """
    Evaluate autoencoder reconstruction fidelity across train, val, and test splits.
    Computes MSE, RMSE, MAE, and R^2 score.
    """
    device = next(model.parameters()).device
    model.eval()
    results = {}

    print(f"\n[AE-RECONSTRUCTION FIDELITY]")
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            X_tensor = data[f"X_{split}"].to(device)
            X_recon, _ = model(X_tensor)

            X_orig_np = X_tensor.cpu().numpy()
            X_recon_np = X_recon.cpu().numpy()

            mse = float(mean_squared_error(X_orig_np, X_recon_np))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(X_orig_np, X_recon_np))
            r2 = float(r2_score(X_orig_np, X_recon_np))

            results[split] = {
                "mse": round(mse, 6),
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "r2_score": round(r2, 6),
                "n_samples": int(len(X_orig_np))
            }
            print(f"  Split: {split:5s} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}")

    return results


def validate_latent_space(latent_train: np.ndarray, latent_val: np.ndarray,
                          y_train: np.ndarray) -> dict:
    """
    Validate the quality of the latent representation.
    
    Checks for:
    - Collapsed dimensions (near-zero variance)
    - NaN/Inf values
    - Extreme values
    - Feature variance distribution
    """
    report = {}
    
    # NaN/Inf check
    report["has_nan"] = bool(np.isnan(latent_train).any())
    report["has_inf"] = bool(np.isinf(latent_train).any())
    
    # Variance per dimension
    variances = np.var(latent_train, axis=0)
    report["dim_variances"] = variances.tolist()
    report["collapsed_dims"] = int(np.sum(variances < 1e-6))
    report["mean_variance"] = float(np.mean(variances))
    report["min_variance"] = float(np.min(variances))
    report["max_variance"] = float(np.max(variances))
    
    # Value range
    report["min_value"] = float(np.min(latent_train))
    report["max_value"] = float(np.max(latent_train))
    report["mean_abs_value"] = float(np.mean(np.abs(latent_train)))
    
    # Print summary
    print(f"\n[LATENT VALIDATION]")
    print(f"  NaN: {report['has_nan']} | Inf: {report['has_inf']}")
    print(f"  Collapsed dims: {report['collapsed_dims']}/{latent_train.shape[1]}")
    print(f"  Variance range: [{report['min_variance']:.6f}, {report['max_variance']:.6f}]")
    print(f"  Value range: [{report['min_value']:.4f}, {report['max_value']:.4f}]")
    
    if report["has_nan"] or report["has_inf"]:
        print("  WARNING: NaN or Inf detected in latent space!")
    if report["collapsed_dims"] > 0:
        print(f"  WARNING: {report['collapsed_dims']} dimensions have near-zero variance!")
    
    return report


def run_autoencoder_pipeline():
    """Full autoencoder training and feature extraction pipeline."""
    print("\n" + "="*60)
    print("AUTOENCODER FEATURE EXTRACTOR")
    print("="*60)
    
    set_seed()
    
    # Load data
    data = load_processed_splits()
    input_dim = data["X_train"].shape[1]
    print(f"[AE] Input dim: {input_dim}, Latent dim: {LATENT_DIM}")
    
    # Create model
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=AE_HIDDEN_DIMS,
        latent_dim=LATENT_DIM
    )
    print(f"[AE] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    history = train_autoencoder(model, data["X_train"], data["X_val"])
    
    # Extract latent features
    latent_train = extract_latent_features(model, data["X_train"])
    latent_val = extract_latent_features(model, data["X_val"])
    latent_test = extract_latent_features(model, data["X_test"])
    
    # Shape validation
    assert latent_train.shape == (len(data["X_train"]), LATENT_DIM), \
        f"Expected ({len(data['X_train'])}, {LATENT_DIM}), got {latent_train.shape}"
    assert latent_val.shape == (len(data["X_val"]), LATENT_DIM)
    assert latent_test.shape == (len(data["X_test"]), LATENT_DIM)
    print(f"[AE] Shape check PASSED: train={latent_train.shape}, "
          f"val={latent_val.shape}, test={latent_test.shape}")
    
    # Validate latent space
    validation_report = validate_latent_space(
        latent_train, latent_val, data["y_train"].numpy()
    )

    # Evaluate reconstruction fidelity
    recon_report = evaluate_reconstruction(model, data)
    
    # Save latent features
    latent_cols = [f"latent_{i}" for i in range(LATENT_DIM)]
    
    pd.DataFrame(latent_train, columns=latent_cols).to_csv(
        os.path.join(RESULTS_DIR, "latent_features_train.csv"), index=False)
    pd.DataFrame(latent_val, columns=latent_cols).to_csv(
        os.path.join(RESULTS_DIR, "latent_features_val.csv"), index=False)
    pd.DataFrame(latent_test, columns=latent_cols).to_csv(
        os.path.join(RESULTS_DIR, "latent_features_test.csv"), index=False)
    print(f"[AE] Latent features saved to {RESULTS_DIR}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "autoencoder.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": AE_HIDDEN_DIMS,
        "latent_dim": LATENT_DIM,
        "history": {
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "training_time": history["training_time"],
        },
        "seed": RANDOM_SEED,
    }, model_path)
    print(f"[AE] Model saved to {model_path}")
    
    # Save training history
    hist_path = os.path.join(RESULTS_DIR, "autoencoder_history.json")
    with open(hist_path, "w") as f:
        json.dump({
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "training_time": history["training_time"],
        }, f, indent=2)
    
    # Save latent validation report
    report_path = os.path.join(RESULTS_DIR, "latent_validation.json")
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)

    # Save reconstruction fidelity report
    recon_path = os.path.join(RESULTS_DIR, "autoencoder_reconstruction.json")
    with open(recon_path, "w") as f:
        json.dump(recon_report, f, indent=2)
    print(f"[AE] Reconstruction report saved to {recon_path}")
    
    return model, history, validation_report, recon_report


# Alias for pipeline runner
train_autoencoder_pipeline = run_autoencoder_pipeline


if __name__ == "__main__":
    model, history, report = run_autoencoder_pipeline()
