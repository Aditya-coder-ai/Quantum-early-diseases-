"""
Part 6: Variational Quantum Classifier (VQC) PyTorch Module.
Encapsulates PennyLane quantum circuit with trainable variational parameters
and calibrated probability output head.
"""
from __future__ import annotations

import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from src.vqc.circuit import create_vqc_circuit, count_quantum_resources


class VariationalQuantumClassifier(nn.Module):
    """
    Hybrid Variational Quantum Classifier module for medical disease classification.
    
    Pipeline:
        Continuous features (Angle-encoded)
                ↓
        Variational Quantum Circuit (Single-qubit rotations + Entanglement)
                ↓
        Expectation Value ⟨Z⟩ ∈ [-1, +1]
                ↓
        Calibrated Linear Head (Scale + Bias)
                ↓
        Logit & Calibrated Probability P(y=1|x)
    """
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        ansatz_type: str = "ansatz_b",
        entanglement: str = "ring",
        measurement_type: str = "expval_z0",
        preferred_device: str = "lightning.qubit",
        fallback_device: str = "default.qubit",
        shots: Optional[int] = None,
        diff_method: str = "adjoint",
        seed: int = 42,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.entanglement = entanglement
        self.measurement_type = measurement_type
        self.seed = seed
        self.shots = shots
        self.diff_method = diff_method

        # Build quantum QNode and resource metadata
        self.qnode, self.device, self.n_params, self.resource_profile = create_vqc_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_type=ansatz_type,
            entanglement=entanglement,
            measurement_type=measurement_type,
            preferred_device=preferred_device,
            fallback_device=fallback_device,
            shots=shots,
            diff_method=diff_method,
        )

        params_per_qubit = self.resource_profile["params_per_qubit"]

        # Controlled parameter initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        init_weights = np.random.uniform(
            -np.pi / 4, np.pi / 4, (n_layers, n_qubits, params_per_qubit)
        ).astype(np.float32)
        self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32, requires_grad=True))

        # Calibrated output head: maps quantum expectation in [-1, +1] to logit space
        # Default initialization: w = -1.0, b = 0.0 maps <Z> = -1 -> logit = +1.0 (P ~ 0.73)
        self.head = nn.Linear(1, 1)
        nn.init.constant_(self.head.weight, -1.0)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward pass.
        
        Args:
            x: Tensor of shape (batch_size, n_qubits) or (n_qubits,)
        Returns:
            logits: Tensor of shape (batch_size,)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Execute quantum QNode
        q_out = self.qnode(x, self.weights)
        
        # Ensure q_out is a PyTorch float32 tensor
        if not isinstance(q_out, torch.Tensor):
            q_out = torch.tensor(q_out, dtype=torch.float32, device=x.device)
        else:
            q_out = q_out.to(dtype=torch.float32)

        if q_out.ndim == 0:
            q_out = q_out.unsqueeze(0)

        # If average_z returned a list of expectations
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out).mean(dim=0)

        # Feed scalar expectation values into calibrated head
        logits = self.head(q_out.unsqueeze(-1)).squeeze(-1)
        return logits

    def predict_proba(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Compute predicted probabilities P(y=1|x) for benign/majority class.
        P(y=0|x) = 1 - P(y=1|x).
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            logits = self.forward(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, x: Union[torch.Tensor, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes in {0, 1}."""
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Save model weights and architectural configuration to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "ansatz_type": self.ansatz_type,
            "entanglement": self.entanglement,
            "measurement_type": self.measurement_type,
            "seed": self.seed,
            "resource_profile": self.resource_profile,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, map_location: str = "cpu") -> VariationalQuantumClassifier:
        """Load trained VQC model from checkpoint file."""
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        model = cls(
            n_qubits=checkpoint["n_qubits"],
            n_layers=checkpoint["n_layers"],
            ansatz_type=checkpoint["ansatz_type"],
            entanglement=checkpoint["entanglement"],
            measurement_type=checkpoint["measurement_type"],
            seed=checkpoint.get("seed", 42),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
