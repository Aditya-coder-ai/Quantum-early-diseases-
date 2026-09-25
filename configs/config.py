"""
Global configuration for MindMatrix project.
All hyperparameters, paths, and seeds are centralized here.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")

# ─── Reproducibility ─────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Data Splitting & Preprocessing Configuration ─────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TARGET_COLUMN = "target"
PATIENT_ID_COLUMN = None   # None for WDBC (each row is unique biopsy sample)
SCALING_METHOD = "standard"  # Options: 'standard', 'minmax', 'robust'
HANDLE_MISSING_STRATEGY = "median"
REMOVE_DUPLICATES = True

# ─── Feature Dimensions ──────────────────────────────────────────────
LATENT_DIM = 16          # Encoder output dimensionality
SELECTED_DIM = 8         # Post quantum-feature-selection dimensionality
NUM_QUBITS = 8           # Number of qubits for VQC

# ─── Autoencoder ──────────────────────────────────────────────────────
AE_HIDDEN_DIMS = [64, 32]  # Encoder hidden layers before latent
AE_LEARNING_RATE = 1e-3
AE_BATCH_SIZE = 32
AE_EPOCHS = 200
AE_PATIENCE = 20          # Early stopping patience

# ─── VQC ──────────────────────────────────────────────────────────────
VQC_NUM_LAYERS = 2        # Number of variational layers
VQC_LEARNING_RATE = 0.01
VQC_EPOCHS = 100
VQC_BATCH_SIZE = 16
VQC_PATIENCE = 15

# ─── Quantum Feature Selection ────────────────────────────────────────
QFS_LAMBDA = 0.1          # Penalty weight for number of features

# Ensure output directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_METADATA_DIR, MODELS_DIR, RESULTS_DIR, CONFIGS_DIR]:
    os.makedirs(d, exist_ok=True)
