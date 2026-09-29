"""
Part 5 — Class Imbalance Handling & Quantum GAN Configuration.

Centralises all hyperparameters, paths, feature sources, and settings for Part 5.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    PROJECT_ROOT, DATA_PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    RANDOM_SEED, SELECTED_DIM
)

# ── Paths ────────────────────────────────────────────────────────────
PART5_RESULTS_DIR = os.path.join(RESULTS_DIR, "part5_imbalance")
PART5_MODELS_DIR = os.path.join(MODELS_DIR, "imbalance")
PART5_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "augmented", "qgan")
PART5_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "imbalance")

# Feature source paths from Part 4
QAOA_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "selected", "qaoa")
CLASSICAL_MI_DIR = os.path.join(PROJECT_ROOT, "features", "selected", "classical_mi")

# Ensure directories exist
for d in [PART5_RESULTS_DIR, PART5_MODELS_DIR, PART5_DATA_DIR, PART5_ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset & Feature Settings ───────────────────────────────────────
FEATURE_DIM = 8  # 8-dimensional compact representation from Part 4
MINORITY_CLASS = 0  # 0 = Malignant in WDBC dataset
MAJORITY_CLASS = 1  # 1 = Benign in WDBC dataset
SEED = RANDOM_SEED

# ── Classical Imbalance Handling ─────────────────────────────────────
# Augmentation ratios: fraction of deficit (majority - minority) to generate
# e.g., 1.0 brings minority count equal to majority count (50/50 balance)
SMOTE_K_NEIGHBORS = 5
AUGMENTATION_RATIOS = [0.25, 0.5, 0.75, 1.0]

# ── Downstream Classifier ────────────────────────────────────────────
DOWNSTREAM_CLASSIFIER = "SVM_RBF"

# ── Quantum GAN Settings ─────────────────────────────────────────────
QGAN_N_QUBITS = FEATURE_DIM  # 8 qubits for 8 features
QGAN_CIRCUIT_DEPTH = 2       # 2 layers of parameterized gates
QGAN_LATENT_DIM = FEATURE_DIM # 8-dimensional noise vector
QGAN_LR_G = 0.01             # Generator learning rate
QGAN_LR_D = 0.005            # Discriminator learning rate
QGAN_EPOCHS = 40             # Training epochs for QGAN
QGAN_BATCH_SIZE = 16         # Mini-batch size
QGAN_D_STEPS = 1             # Discriminator steps per generator step
QGAN_STABILITY_SEEDS = [42, 43, 44] # Seeds for multi-run stability testing
