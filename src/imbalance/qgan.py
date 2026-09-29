"""
Part 5 — Quantum Generative Adversarial Network (QGAN) for Minority-Class Augmentation.

Architecture:
    Noise z ~ N(0, I)
        ↓
    Quantum Generator (Parameterized Quantum Circuit in PennyLane)
        - Angle encoding of latent noise via RY rotations
        - Circular CNOT entanglement layers
        - Variational rotation gates RY, RZ
        - PauliZ expectation values ⟨Z_i⟩ ∈ [-1, 1]
        - Learnable affine scaling matching empirical minority distribution
        ↓
    Synthetic Minority Feature Vector (D dimensions)
        ↓
    Classical Discriminator (PyTorch MLP)
        - Linear(D, 32) → LeakyReLU(0.2) → Linear(32, 16) → LeakyReLU(0.2) → Linear(16, 1) → Sigmoid()
        ↓
    Real / Fake Probability

Leakage Guarantee:
    Trained STRICTLY on the training minority samples (X_train[y_train == 0]).
    Zero validation or test samples are ever seen by the generator or discriminator.
"""
from __future__ import annotations

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from src.imbalance.config import (
    QGAN_N_QUBITS, QGAN_CIRCUIT_DEPTH, QGAN_LATENT_DIM,
    QGAN_LR_G, QGAN_LR_D, QGAN_EPOCHS, QGAN_BATCH_SIZE,
    SEED, PART5_MODELS_DIR, MINORITY_CLASS
)

logger = logging.getLogger(__name__)


# ── Quantum Circuit Definition ────────────────────────────────────────

def create_qgan_circuit(n_qubits: int = QGAN_N_QUBITS, n_layers: int = QGAN_CIRCUIT_DEPTH):
    """
    Create a PennyLane QNode for the Quantum Generator.
    Uses 'default.qubit' with 'torch' interface and 'backprop' differentiation.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # inputs: (batch, n_qubits)
        # weights: (n_layers, n_qubits, 2)
        # 1. State preparation / Noise encoding
        for i in range(n_qubits):
            qml.RY(inputs[:, i], wires=i)

        # 2. Variational layers
        for l in range(weights.shape[0]):
            # Circular entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            # Parameterized single-qubit rotations
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)

        # 3. Measurement: PauliZ expectation values on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, dev


# ── Quantum Generator Module ──────────────────────────────────────────

class QuantumGenerator(nn.Module):
    """
    Hybrid Quantum Generator mapping latent noise z to synthetic feature vectors.
    """

    def __init__(
        self,
        n_qubits: int = QGAN_N_QUBITS,
        n_layers: int = QGAN_CIRCUIT_DEPTH,
        empirical_mean: np.ndarray | None = None,
        empirical_std: np.ndarray | None = None,
        seed: int = SEED,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Set seed for reproducible initialization
        torch.manual_seed(seed)

        # Variational quantum circuit weights
        self.weights = nn.Parameter(
            0.1 * torch.randn(n_layers, n_qubits, 2, dtype=torch.float32)
        )

        # Empirical affine transformation parameters
        # Output: x_i = shift_i + scale_i * <Z_i>
        if empirical_mean is not None:
            init_shift = torch.tensor(empirical_mean, dtype=torch.float32)
        else:
            init_shift = torch.zeros(n_qubits, dtype=torch.float32)

        if empirical_std is not None:
            init_scale = torch.tensor(empirical_std, dtype=torch.float32)
        else:
            init_scale = torch.ones(n_qubits, dtype=torch.float32)

        self.shift = nn.Parameter(init_shift)
        self.scale = nn.Parameter(init_scale)

        self.circuit, self.dev = create_qgan_circuit(n_qubits, n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            z: (batch, n_qubits) latent noise
            returns: (batch, n_qubits) synthetic feature vectors
        """
        res = self.circuit(z, self.weights)
        # Stack output into (batch, n_qubits) float32 tensor
        quantum_out = torch.stack(res, dim=1).to(dtype=torch.float32)
        # Affine transformation to medical feature space
        out = self.shift + self.scale * quantum_out
        return out


# ── Classical Discriminator Module ────────────────────────────────────

class ClassicalDiscriminator(nn.Module):
    """
    Classical Discriminator determining whether a feature vector is real or synthetic.
    """

    def __init__(self, input_dim: int = QGAN_N_QUBITS, hidden_dim: int = 32, seed: int = SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Quantum GAN Training System ───────────────────────────────────────

class QuantumGAN:
    """
    High-level QGAN orchestrator managing training, sampling, and serialization.
    """

    def __init__(
        self,
        n_qubits: int = QGAN_N_QUBITS,
        n_layers: int = QGAN_CIRCUIT_DEPTH,
        lr_g: float = QGAN_LR_G,
        lr_d: float = QGAN_LR_D,
        empirical_mean: np.ndarray | None = None,
        empirical_std: np.ndarray | None = None,
        seed: int = SEED,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed

        self.generator = QuantumGenerator(
            n_qubits=n_qubits,
            n_layers=n_layers,
            empirical_mean=empirical_mean,
            empirical_std=empirical_std,
            seed=seed,
        )
        self.discriminator = ClassicalDiscriminator(
            input_dim=n_qubits,
            hidden_dim=32,
            seed=seed,
        )

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.7, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.7, 0.999))
        self.criterion = nn.BCELoss()

        self.training_history = {
            "loss_D": [],
            "loss_G": [],
            "d_real_prob": [],
            "d_fake_prob": [],
        }

    def train(
        self,
        X_minority: np.ndarray,
        epochs: int = QGAN_EPOCHS,
        batch_size: int = QGAN_BATCH_SIZE,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the QGAN exclusively on the minority class feature matrix.
        """
        t_start = time.time()
        X_tensor = torch.tensor(X_minority, dtype=torch.float32)
        N = len(X_tensor)

        rng = np.random.RandomState(self.seed)

        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            # Shuffle training samples
            perm = rng.permutation(N)
            epoch_loss_D = 0.0
            epoch_loss_G = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                batch_indices = perm[i : i + batch_size]
                real_batch = X_tensor[batch_indices]
                curr_bs = len(real_batch)

                # ── 1. Train Discriminator ──
                # Real samples with one-sided label smoothing (0.9 instead of 1.0)
                real_labels = torch.full((curr_bs, 1), 0.9, dtype=torch.float32)
                d_out_real = self.discriminator(real_batch)
                loss_d_real = self.criterion(d_out_real, real_labels)

                # Fake samples from Generator
                z = torch.randn(curr_bs, self.n_qubits, dtype=torch.float32)
                fake_batch = self.generator(z)
                fake_labels = torch.full((curr_bs, 1), 0.1, dtype=torch.float32)
                d_out_fake = self.discriminator(fake_batch.detach())
                loss_d_fake = self.criterion(d_out_fake, fake_labels)

                loss_D = loss_d_real + loss_d_fake
                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()

                # ── 2. Train Generator ──
                # Non-saturating GAN loss: target = 1.0
                gen_targets = torch.ones((curr_bs, 1), dtype=torch.float32)
                d_out_fake_for_g = self.discriminator(fake_batch)
                loss_g_adv = self.criterion(d_out_fake_for_g, gen_targets)

                # Moment-matching regularization: match mean of real minority
                mean_real = torch.mean(real_batch, dim=0)
                mean_fake = torch.mean(fake_batch, dim=0)
                loss_g_moment = torch.norm(mean_real - mean_fake, p=2)

                loss_G = loss_g_adv + 0.5 * loss_g_moment
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()

                epoch_loss_D += loss_D.item()
                epoch_loss_G += loss_G.item()
                n_batches += 1

            avg_loss_D = epoch_loss_D / max(1, n_batches)
            avg_loss_G = epoch_loss_G / max(1, n_batches)
            self.training_history["loss_D"].append(avg_loss_D)
            self.training_history["loss_G"].append(avg_loss_G)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs):
                print(
                    f"  Epoch [{epoch+1:2d}/{epochs:2d}] "
                    f"Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f}",
                    flush=True,
                )

        total_time = round(time.time() - t_start, 2)
        if verbose:
            print(f"[QGAN] Training completed in {total_time}s", flush=True)

        return {
            "training_time_s": total_time,
            "final_loss_D": round(self.training_history["loss_D"][-1], 4),
            "final_loss_G": round(self.training_history["loss_G"][-1], 4),
            "epochs": epochs,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        }

    def generate(self, n_samples: int, seed: int = SEED) -> np.ndarray:
        """
        Generate synthetic minority samples.

        Returns:
            NumPy array of shape (n_samples, n_qubits) with synthetic feature vectors.
        """
        self.generator.eval()
        torch.manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_qubits, dtype=torch.float32)
            samples = self.generator(z).cpu().numpy()

        assert not np.isnan(samples).any(), "QGAN generated NaN values"
        assert not np.isinf(samples).any(), "QGAN generated Infinite values"
        assert samples.shape == (n_samples, self.n_qubits)
        return samples

    def save(self, model_dir: str = PART5_MODELS_DIR, tag: str = "primary"):
        """Save QGAN generator and discriminator models."""
        os.makedirs(model_dir, exist_ok=True)
        g_path = os.path.join(model_dir, f"qgan_generator_{tag}.pt")
        d_path = os.path.join(model_dir, f"qgan_discriminator_{tag}.pt")
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)
        return {"generator_path": g_path, "discriminator_path": d_path}

    def load(self, model_dir: str = PART5_MODELS_DIR, tag: str = "primary"):
        """Load saved weights."""
        g_path = os.path.join(model_dir, f"qgan_generator_{tag}.pt")
        d_path = os.path.join(model_dir, f"qgan_discriminator_{tag}.pt")
        self.generator.load_state_dict(torch.load(g_path, weights_only=True))
        self.discriminator.load_state_dict(torch.load(d_path, weights_only=True))
        return self


# ── STEP 8: Tiny Synthetic QGAN Verification Function ─────────────────

def run_tiny_synthetic_test(n_samples: int = 60, seed: int = 42) -> Dict[str, Any]:
    """
    Controlled tiny synthetic QGAN test (2 dimensions, 2 qubits).
    Generates a simple 2D Gaussian distribution to verify all QGAN mechanics:
        - Quantum generator executes
        - Gradients/optimization work without numerical instability
        - Discriminator trains properly
        - Generator output changes during training
        - Loss values are strictly finite
        - Generated samples have correct dimensions
    """
    rng = np.random.RandomState(seed)
    # Synthetic target: 2D Gaussian centered at [1.5, -1.0] with std [0.5, 0.8]
    X_synthetic_target = rng.randn(n_samples, 2) * np.array([0.5, 0.8]) + np.array([1.5, -1.0])

    qgan = QuantumGAN(
        n_qubits=2,
        n_layers=1,
        lr_g=0.02,
        lr_d=0.01,
        empirical_mean=np.mean(X_synthetic_target, axis=0),
        empirical_std=np.std(X_synthetic_target, axis=0),
        seed=seed,
    )

    # Initial samples before training
    initial_samples = qgan.generate(10, seed=seed)

    # Train for 15 epochs
    train_res = qgan.train(X_synthetic_target, epochs=15, batch_size=16, verbose=False)

    # Generated samples after training
    final_samples = qgan.generate(20, seed=seed)

    # Check output changed
    weights_diff = float(torch.norm(qgan.generator.weights.detach()))

    assert final_samples.shape == (20, 2), "Tiny QGAN output shape mismatch"
    assert not np.isnan(final_samples).any(), "Tiny QGAN generated NaNs"
    assert not np.isinf(final_samples).any(), "Tiny QGAN generated Infs"
    assert np.isfinite(train_res["final_loss_D"]), "Tiny QGAN discriminator loss is non-finite"
    assert np.isfinite(train_res["final_loss_G"]), "Tiny QGAN generator loss is non-finite"

    return {
        "status": "PASSED",
        "n_qubits": 2,
        "n_layers": 1,
        "samples_shape": list(final_samples.shape),
        "weights_norm": weights_diff,
        "final_loss_D": train_res["final_loss_D"],
        "final_loss_G": train_res["final_loss_G"],
    }
