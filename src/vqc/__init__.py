"""
Part 6: Variational Quantum Classifier (VQC) Package.
Combines feature encoding, quantum variational circuits, training,
evaluation, noise analysis, and ablation experiments.
"""
from src.vqc.config import VQCConfig
from src.vqc.encoding import AngleScaler, FeatureQubitMapper
from src.vqc.circuit import create_vqc_circuit, count_quantum_resources
from src.vqc.model import VariationalQuantumClassifier

__all__ = [
    "VQCConfig",
    "AngleScaler",
    "FeatureQubitMapper",
    "create_vqc_circuit",
    "count_quantum_resources",
    "VariationalQuantumClassifier",
]
