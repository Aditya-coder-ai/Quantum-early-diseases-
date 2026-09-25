"""
Diagnostic script to benchmark VQC execution and gradient calculation.
Tests both autograd and PyTorch interfaces, and default.qubit vs lightning.qubit.
"""
import time
import numpy as np
import torch
import pennylane as qml
from configs.config import NUM_QUBITS, VQC_NUM_LAYERS, RANDOM_SEED

print("--- Testing PennyLane Devices & Gradients ---")

# 1. Test device
device_name = "lightning.qubit"
try:
    dev = qml.device(device_name, wires=NUM_QUBITS)
    print(f"Successfully initialized {device_name}")
except Exception as e:
    print(f"Failed {device_name}, falling back to default.qubit: {e}")
    device_name = "default.qubit"
    dev = qml.device(device_name, wires=NUM_QUBITS)

# 2. Test Torch interface with adjoint diff
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit_torch(inputs, weights):
    for i in range(NUM_QUBITS):
        qml.RY(inputs[i], wires=i)
    for l in range(VQC_NUM_LAYERS):
        for i in range(NUM_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(NUM_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
    return qml.expval(qml.PauliZ(0))

x = torch.tensor(np.random.uniform(0, np.pi, NUM_QUBITS), dtype=torch.float32)
w = torch.tensor(np.random.uniform(-0.5, 0.5, (VQC_NUM_LAYERS, NUM_QUBITS, 2)), dtype=torch.float32, requires_grad=True)

t0 = time.time()
out = circuit_torch(x, w)
out.backward()
dt = time.time() - t0
print(f"Single sample forward+backward (adjoint + torch): {dt*1000:.2f} ms")
print(f"Grad norm: {w.grad.norm().item():.4f}")

# Test batch of 16 using a simple forward pass
t0 = time.time()
batch_x = torch.tensor(np.random.uniform(0, np.pi, (16, NUM_QUBITS)), dtype=torch.float32)
w.grad.zero_()
loss = 0.0
for i in range(16):
    exp_val = circuit_torch(batch_x[i], w)
    prob = (1.0 - exp_val) / 2.0
    loss = loss + (prob - 1.0)**2
loss = loss / 16
loss.backward()
dt_batch = time.time() - t0
print(f"Batch of 16 forward+backward: {dt_batch*1000:.2f} ms")
print(f"Estimated time per epoch (25 batches): {dt_batch * 25:.2f} s")
