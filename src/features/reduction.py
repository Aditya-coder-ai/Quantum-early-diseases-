"""
Feature reduction and representation learning module for Part 3.

Implements zero-leakage dimensionality reduction using:
1. Principal Component Analysis (PCA)
2. PyTorch Neural Autoencoder (AE)

Supports target dimensions: 4, 8, 12, 16.
All reducers fit EXCLUSIVELY on training data (X_train).
Validation (X_val) and test (X_test) data are transformed using training-learned parameters.
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, MODELS_DIR, RESULTS_DIR, DATA_PROCESSED_DIR
)


def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for numpy and PyTorch reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PCAReducer:
    """
    Leakage-safe PCA feature reducer.
    Fits strictly on training features and transforms val/test splits.
    """
    def __init__(self, n_components: int = 8, random_state: int = RANDOM_SEED):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False
        self.explained_variance_ratio_: np.ndarray = np.array([])
        self.total_explained_variance_: float = 0.0

    def fit(self, X_train: pd.DataFrame | np.ndarray) -> "PCAReducer":
        """Fit PCA strictly on training dataset."""
        X_mat = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.pca.fit(X_mat)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.total_explained_variance_ = float(np.sum(self.explained_variance_ratio_))
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using training-fitted PCA."""
        if not self.is_fitted:
            raise RuntimeError("PCAReducer must be fitted before calling transform().")
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        features = self.pca.transform(X_mat)
        assert not np.isnan(features).any(), "NaN detected in PCA output."
        assert not np.isinf(features).any(), "Inf detected in PCA output."
        return features

    def fit_transform(self, X_train: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit on train and transform train."""
        return self.fit(X_train).transform(X_train)

    def save(self, filepath: str):
        """Save fitted PCA object to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "PCAReducer":
        """Load fitted PCA object from disk."""
        reducer = joblib.load(filepath)
        if not isinstance(reducer, PCAReducer):
            raise TypeError("Loaded object is not a PCAReducer instance.")
        return reducer


class ConfigurableAutoencoder(nn.Module):
    """
    PyTorch Autoencoder with configurable latent dimension.
    Encoder: input_dim -> 64 -> 32 -> latent_dim
    Decoder: latent_dim -> 32 -> 64 -> input_dim
    """
    def __init__(self, input_dim: int = 30, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder architecture
        if input_dim >= 30:
            hidden_dims = [64, 32]
        else:
            hidden_dims = [max(16, input_dim // 2)]

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

        # Decoder architecture (mirror)
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
        """Encode input into latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector into reconstructed space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, latent)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoencoderReducer:
    """
    Wrapper for Autoencoder training, latent feature extraction, and persistence.
    """
    def __init__(
        self,
        latent_dim: int = 8,
        input_dim: int = 30,
        epochs: int = 150,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 20,
        random_seed: int = RANDOM_SEED
    ):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_seed = random_seed

        self.model = ConfigurableAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.is_fitted = False
        self.history: Dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame | np.ndarray, X_val: pd.DataFrame | np.ndarray) -> "AutoencoderReducer":
        """Train autoencoder on X_train with early stopping on X_val."""
        set_seed(self.random_seed)
        
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val

        X_tr_t = torch.FloatTensor(X_train_np)
        X_va_t = torch.FloatTensor(X_val_np)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        criterion = nn.MSELoss()

        train_loader = DataLoader(TensorDataset(X_tr_t), batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        train_losses, val_losses = [], []
        t0 = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            b_losses = []
            for (bx,) in train_loader:
                bx = bx.to(device)
                x_recon, _ = self.model(bx)
                loss = criterion(x_recon, bx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                b_losses.append(loss.item())

            avg_tr_loss = float(np.mean(b_losses))
            train_losses.append(avg_tr_loss)

            self.model.eval()
            with torch.no_grad():
                X_val_dev = X_va_t.to(device)
                val_recon, _ = self.model(X_val_dev)
                val_loss = float(criterion(val_recon, X_val_dev).item())
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        training_time = time.time() - t0
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True

        self.history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_time": round(training_time, 4),
        }
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract latent representation using fitted encoder."""
        if not self.is_fitted:
            raise RuntimeError("AutoencoderReducer must be fitted before calling transform().")
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        X_t = torch.FloatTensor(X_np)
        
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(X_t.to(device))
        res = z.cpu().numpy()
        assert not np.isnan(res).any(), "NaN detected in Autoencoder latent output."
        assert not np.isinf(res).any(), "Inf detected in Autoencoder latent output."
        return res

    def save(self, filepath: str):
        """Save fitted Autoencoder checkpoint and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "history": self.history,
            "seed": self.random_seed,
        }, filepath)

    @staticmethod
    def load(filepath: str) -> "AutoencoderReducer":
        """Load fitted Autoencoder from checkpoint."""
        checkpoint = torch.load(filepath, weights_only=False)
        reducer = AutoencoderReducer(
            latent_dim=checkpoint["latent_dim"],
            input_dim=checkpoint["input_dim"],
            random_seed=checkpoint["seed"]
        )
        reducer.model.load_state_dict(checkpoint["model_state_dict"])
        reducer.model.eval()
        reducer.is_fitted = True
        reducer.history = checkpoint.get("history", {})
        return reducer
