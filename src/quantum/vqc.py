"""
Variational Quantum Circuit (VQC) for binary classification.

Architecture:
    8 qubits
    Angle encoding (RY gates)
    Trainable rotation layers (RY, RZ)
    Entanglement layers (circular CNOT chain)
    Measurement (PauliZ on qubit 0) -> scalar expectation value
    Classical trainable scale + bias (linear head) -> calibrated probability

Optimized with:
- PennyLane PyTorch interface
- Adjoint differentiation on lightning.qubit (with default.qubit fallback)
- Broadcasted batch execution for high computational throughput
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import NUM_QUBITS, VQC_NUM_LAYERS, RANDOM_SEED


class AngleScaler:
    """
    Fit angle normalization on training data only; apply to any split.
    Prevents data leakage in quantum encoding.
    Maps feature values to [0, pi] for RY rotation gates.
    """
    def __init__(self):
        self.min_vals = None
        self.range_vals = None

    def fit(self, X_train: np.ndarray):
        """Fit min/max per feature from training data."""
        self.min_vals = np.asarray(X_train).min(axis=0)
        max_vals = np.asarray(X_train).max(axis=0)
        self.range_vals = max_vals - self.min_vals
        self.range_vals[self.range_vals < 1e-10] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to angle range [0, pi]."""
        X_arr = np.asarray(X, dtype=np.float32)
        X_normalized = (X_arr - self.min_vals) / self.range_vals
        X_normalized = np.clip(X_normalized, 0.0, 1.0)
        return (X_normalized * np.pi).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)


def get_quantum_device(n_qubits: int = NUM_QUBITS):
    """
    Initialize quantum simulator device.
    Prefers high-performance 'lightning.qubit', falls back to 'default.qubit'.
    """
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
    return dev


def create_vqc(
    n_qubits: int = NUM_QUBITS,
    n_layers: int = VQC_NUM_LAYERS,
    device_name: str | None = None
) -> tuple:
    """
    Create a VQC circuit and its QNode with PyTorch interface and adjoint differentiation.
    
    Returns:
        (circuit_fn, dev, n_params) tuple
    """
    if device_name:
        dev = qml.device(device_name, wires=n_qubits)
    else:
        dev = get_quantum_device(n_qubits)
        
    n_params = n_layers * n_qubits * 2

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        """
        inputs: shape (batch_size, n_qubits) or (n_qubits,)
        weights: shape (n_layers, n_qubits, 2)
        """
        # Data encoding layer (RY rotations)
        if inputs.ndim == 1:
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
        else:
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)

        # Variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # Entanglement (circular CNOT chain)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        return qml.expval(qml.PauliZ(0))

    return circuit, dev, n_params


def init_weights(
    n_layers: int = VQC_NUM_LAYERS,
    n_qubits: int = NUM_QUBITS,
    seed: int = RANDOM_SEED
) -> torch.Tensor:
    """Initialize VQC weights with uniform distribution in [-pi/4, pi/4]."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    w = np.random.uniform(-np.pi / 4, np.pi / 4, (n_layers, n_qubits, 2)).astype(np.float32)
    return torch.tensor(w, dtype=torch.float32, requires_grad=True)


class HybridVQCModule(nn.Module):
    """
    PyTorch Hybrid Quantum-Classical Module.
    Combines:
    1. Variational Quantum Circuit (8 qubits, 2 layers)
    2. Classical linear scaling layer (expectation value -> logit)
    """
    def __init__(
        self,
        n_qubits: int = NUM_QUBITS,
        n_layers: int = VQC_NUM_LAYERS,
        seed: int = RANDOM_SEED
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit, self.dev, self.n_params = create_vqc(n_qubits, n_layers)

        torch.manual_seed(seed)
        init_w = np.random.uniform(-np.pi / 4, np.pi / 4, (n_layers, n_qubits, 2)).astype(np.float32)
        self.weights = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))

        # Linear head maps quantum expectation in [-1, 1] to logit
        self.head = nn.Linear(1, 1)
        nn.init.constant_(self.head.weight, -1.0)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input angles (batch_size, n_qubits)
        Returns:
            logits: (batch_size,)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        # Quantum expectation values: shape (batch_size,)
        q_out = self.circuit(x, self.weights)
        if q_out.ndim == 0:
            q_out = q_out.unsqueeze(0)
            
        # Reshape to (batch_size, 1) for linear head
        logits = self.head(q_out.unsqueeze(-1)).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Predict class probabilities P(y=1|x)."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            logits = self.forward(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, x: torch.Tensor | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes {0, 1}."""
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)


def vqc_predict_batch(
    model: HybridVQCModule,
    X_angles: np.ndarray,
    batch_size: int = 64
) -> np.ndarray:
    """Predict probabilities for a batch of samples."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_angles), batch_size):
            end = min(start + batch_size, len(X_angles))
            batch = torch.tensor(X_angles[start:end], dtype=torch.float32)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)
