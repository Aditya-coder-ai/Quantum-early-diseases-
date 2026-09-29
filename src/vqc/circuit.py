"""
Part 6: Variational Quantum Circuits, Ansätze, and Resource Estimation.
Constructs differentiable PennyLane QNodes and counts exact quantum resources.
"""
from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Tuple, Dict, Any, Optional


def count_quantum_resources(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str = "ansatz_b",
    entanglement: str = "ring"
) -> Dict[str, Any]:
    """
    Analytically compute the quantum resource profile for the configured VQC architecture.
    
    Args:
        n_qubits: Number of quantum wires / qubits
        n_layers: Number of variational repetitions
        ansatz_type: 'ansatz_a' (RY only) or 'ansatz_b' (RY + RZ)
        entanglement: 'linear' (N-1 CNOTs) or 'ring' (N CNOTs)
    """
    encoding_gates = n_qubits  # 1 RY per qubit for angle encoding
    
    # Rotation gates per layer
    if ansatz_type == "ansatz_a":
        rotations_per_layer = n_qubits
        params_per_qubit = 1
    elif ansatz_type == "ansatz_b":
        rotations_per_layer = 2 * n_qubits
        params_per_qubit = 2
    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type}")

    # Entangling gates per layer
    if entanglement == "linear":
        cnots_per_layer = max(0, n_qubits - 1)
    elif entanglement == "ring":
        cnots_per_layer = n_qubits if n_qubits > 1 else 0
    else:
        raise ValueError(f"Unknown entanglement pattern: {entanglement}")

    total_single_qubit_gates = encoding_gates + (n_layers * rotations_per_layer)
    total_entangling_gates = n_layers * cnots_per_layer
    total_gates = total_single_qubit_gates + total_entangling_gates
    total_trainable_params = n_layers * n_qubits * params_per_qubit

    # Estimated circuit depth:
    # 1 for encoding + n_layers * (rotations_depth + entanglement_depth)
    rot_depth = 1 if ansatz_type == "ansatz_a" else 2
    cnot_depth = 2 if entanglement == "ring" else 2
    estimated_depth = 1 + n_layers * (rot_depth + cnot_depth)

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "ansatz_type": ansatz_type,
        "entanglement": entanglement,
        "encoding_gates": encoding_gates,
        "rotations_per_layer": rotations_per_layer,
        "cnots_per_layer": cnots_per_layer,
        "total_single_qubit_gates": total_single_qubit_gates,
        "total_entangling_gates": total_entangling_gates,
        "total_gates": total_gates,
        "circuit_depth": estimated_depth,
        "trainable_parameters": total_trainable_params,
        "params_per_qubit": params_per_qubit,
    }


def get_quantum_device(
    n_qubits: int,
    preferred_device: str = "lightning.qubit",
    fallback_device: str = "default.qubit",
    shots: Optional[int] = None
) -> Tuple[qml.Device, str]:
    """
    Initialize quantum simulator device.
    Attempts preferred high-performance device, falling back if unavailable.
    """
    device_used = preferred_device
    try:
        dev = qml.device(preferred_device, wires=n_qubits, shots=shots)
    except Exception:
        device_used = fallback_device
        dev = qml.device(fallback_device, wires=n_qubits, shots=shots)
    return dev, device_used


def create_vqc_circuit(
    n_qubits: int,
    n_layers: int = 2,
    ansatz_type: str = "ansatz_b",
    entanglement: str = "ring",
    measurement_type: str = "expval_z0",
    preferred_device: str = "lightning.qubit",
    fallback_device: str = "default.qubit",
    shots: Optional[int] = None,
    diff_method: str = "adjoint"
):
    """
    Constructs an optimized, differentiable PennyLane QNode for binary classification.
    
    Args:
        n_qubits: Number of qubits (equal to input feature dimension)
        n_layers: Number of variational layers
        ansatz_type: 'ansatz_a' or 'ansatz_b'
        entanglement: 'linear' or 'ring'
        measurement_type: 'expval_z0' or 'average_z'
        diff_method: 'adjoint' or 'backprop'
    
    Returns:
        (qnode, device, n_params, resource_profile)
    """
    # If shots are enabled or using noisy device, adjoint diff is not supported -> use parameter-shift or finite-diff
    effective_diff = diff_method
    if shots is not None and effective_diff == "adjoint":
        effective_diff = "parameter-shift"

    dev, dev_name = get_quantum_device(
        n_qubits, preferred_device=preferred_device, fallback_device=fallback_device, shots=shots
    )
    resources = count_quantum_resources(n_qubits, n_layers, ansatz_type, entanglement)
    params_per_qubit = resources["params_per_qubit"]

    @qml.qnode(dev, interface="torch", diff_method=effective_diff)
    def qnode(inputs, weights):
        """
        Quantum variational circuit execution.
        Args:
            inputs: Tensor of angle-encoded features of shape (batch_size, n_qubits) or (n_qubits,)
            weights: Tensor of variational parameters of shape (n_layers, n_qubits, params_per_qubit)
        Returns:
            Scalar or batch expectation value in [-1, +1]
        """
        # 1. Feature Encoding Layer (RY rotations)
        if inputs.ndim == 1:
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
        else:
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)

        # 2. Variational Ansatz Layers
        for l in range(n_layers):
            # Parameterized Single-Qubit Rotations
            for i in range(n_qubits):
                if ansatz_type == "ansatz_a":
                    qml.RY(weights[l, i, 0], wires=i)
                elif ansatz_type == "ansatz_b":
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)

            # Entanglement Layer
            if entanglement == "linear":
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            elif entanglement == "ring":
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # 3. Measurement
        if measurement_type == "expval_z0":
            return qml.expval(qml.PauliZ(0))
        elif measurement_type == "average_z":
            # Multi-qubit average Pauli Z measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        else:
            raise ValueError(f"Unknown measurement_type: {measurement_type}")

    return qnode, dev, resources["trainable_parameters"], resources
