"""
Part 6: Quantum Noise Modeling & Simulation.
Evaluates trained VQC robustness against finite-shot noise and mixed-state depolarizing noise.
"""
from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Dict, Any, Optional

from src.vqc.evaluation import compute_comprehensive_metrics


def build_noisy_vqc_qnode(
    n_qubits: int,
    n_layers: int = 2,
    ansatz_type: str = "ansatz_b",
    entanglement: str = "ring",
    noise_type: str = "shots",
    shots: int = 1024,
    depolarizing_prob: float = 0.01,
):
    """
    Builds a noisy QNode using finite sampling shots or mixed-state depolarizing channels.
    
    Args:
        n_qubits: Number of quantum wires
        n_layers: Variational depth
        ansatz_type: 'ansatz_a' or 'ansatz_b'
        entanglement: 'linear' or 'ring'
        noise_type: 'shots' or 'depolarizing'
        shots: Number of measurement shots for shot noise
        depolarizing_prob: Error probability per gate for depolarizing noise
    """
    if noise_type == "shots":
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    elif noise_type == "depolarizing":
        dev = qml.device("default.mixed", wires=n_qubits)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    @qml.qnode(dev, interface="torch")
    def noisy_qnode(inputs, weights):
        # 1. Angle encoding
        if inputs.ndim == 1:
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
        else:
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)

        # Optional depolarizing channel after encoding
        if noise_type == "depolarizing":
            for i in range(n_qubits):
                qml.DepolarizingChannel(depolarizing_prob, wires=i)

        # 2. Variational layers
        for l in range(n_layers):
            for i in range(n_qubits):
                if ansatz_type == "ansatz_a":
                    qml.RY(weights[l, i, 0], wires=i)
                elif ansatz_type == "ansatz_b":
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)

            if noise_type == "depolarizing":
                for i in range(n_qubits):
                    qml.DepolarizingChannel(depolarizing_prob, wires=i)

            if entanglement == "linear":
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            elif entanglement == "ring":
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            if noise_type == "depolarizing":
                for i in range(n_qubits):
                    qml.DepolarizingChannel(depolarizing_prob, wires=i)

        return qml.expval(qml.PauliZ(0))

    return noisy_qnode


def evaluate_under_noise(
    trained_model,
    X_angles: np.ndarray,
    y_true: np.ndarray,
    shots: int = 1024,
    depolarizing_prob: float = 0.01,
    minority_class: int = 0
) -> Dict[str, Any]:
    """
    Evaluates a trained VQC model under both finite-shot noise and depolarizing noise.
    """
    trained_weights = trained_model.weights.detach()
    head = trained_model.head
    head.eval()

    n_qubits = trained_model.n_qubits
    n_layers = trained_model.n_layers
    ansatz_type = trained_model.ansatz_type
    entanglement = trained_model.entanglement

    results = {}

    # 1. Ideal Baseline
    ideal_probs = trained_model.predict_proba(X_angles)
    ideal_metrics = compute_comprehensive_metrics(
        y_true, (ideal_probs >= 0.5).astype(int), ideal_probs, minority_class=minority_class
    )
    results["ideal"] = ideal_metrics

    # 2. Shot Noise (shots=1024)
    shot_qnode = build_noisy_vqc_qnode(
        n_qubits=n_qubits, n_layers=n_layers, ansatz_type=ansatz_type,
        entanglement=entanglement, noise_type="shots", shots=shots
    )

    shot_probs = []
    with torch.no_grad():
        for i in range(len(X_angles)):
            x_i = torch.tensor(X_angles[i], dtype=torch.float32)
            q_out = shot_qnode(x_i, trained_weights)
            if not isinstance(q_out, torch.Tensor):
                q_out = torch.tensor(q_out, dtype=torch.float32)
            else:
                q_out = q_out.to(dtype=torch.float32)
            logit = head(q_out.unsqueeze(-1)).squeeze(-1)
            shot_probs.append(float(torch.sigmoid(logit).item()))

    shot_probs_arr = np.array(shot_probs)
    shot_metrics = compute_comprehensive_metrics(
        y_true, (shot_probs_arr >= 0.5).astype(int), shot_probs_arr, minority_class=minority_class
    )
    shot_metrics["prob_mae_vs_ideal"] = round(float(np.mean(np.abs(shot_probs_arr - ideal_probs))), 4)
    shot_metrics["shots"] = shots
    results["shot_noise"] = shot_metrics

    # 3. Depolarizing Noise
    depol_qnode = build_noisy_vqc_qnode(
        n_qubits=n_qubits, n_layers=n_layers, ansatz_type=ansatz_type,
        entanglement=entanglement, noise_type="depolarizing", depolarizing_prob=depolarizing_prob
    )

    depol_probs = []
    with torch.no_grad():
        for i in range(len(X_angles)):
            x_i = torch.tensor(X_angles[i], dtype=torch.float32)
            q_out = depol_qnode(x_i, trained_weights)
            if not isinstance(q_out, torch.Tensor):
                q_out = torch.tensor(q_out, dtype=torch.float32)
            else:
                q_out = q_out.to(dtype=torch.float32)
            logit = head(q_out.unsqueeze(-1)).squeeze(-1)
            depol_probs.append(float(torch.sigmoid(logit).item()))

    depol_probs_arr = np.array(depol_probs)
    depol_metrics = compute_comprehensive_metrics(
        y_true, (depol_probs_arr >= 0.5).astype(int), depol_probs_arr, minority_class=minority_class
    )
    depol_metrics["prob_mae_vs_ideal"] = round(float(np.mean(np.abs(depol_probs_arr - ideal_probs))), 4)
    depol_metrics["depolarizing_prob"] = depolarizing_prob
    results["depolarizing_noise"] = depol_metrics

    return results
