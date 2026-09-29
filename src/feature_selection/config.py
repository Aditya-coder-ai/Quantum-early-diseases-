"""
Part 4 — Feature Selection Configuration.

All hyperparameters, paths, budgets, and QAOA settings centralised here.
Imports the project-wide config and extends it with Part-4 specifics.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    PROJECT_ROOT, DATA_PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    RANDOM_SEED, SELECTED_DIM, QFS_LAMBDA
)

# ── Paths ────────────────────────────────────────────────────────────
FEATURES_CLASSICAL_DIR = os.path.join(PROJECT_ROOT, "features", "classical")
LATENT_DIR = RESULTS_DIR  # latent_features_{split}.csv live here

PART4_RESULTS_DIR = os.path.join(RESULTS_DIR, "part4_feature_selection")
PART4_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "features", "selected")

# Sub-dirs per method
ARTIFACT_DIRS = {
    "mutual_info": os.path.join(PART4_ARTIFACTS_DIR, "classical_mi"),
    "rfe": os.path.join(PART4_ARTIFACTS_DIR, "classical_rfe"),
    "lasso": os.path.join(PART4_ARTIFACTS_DIR, "classical_lasso"),
    "rf_importance": os.path.join(PART4_ARTIFACTS_DIR, "classical_rf"),
    "qaoa": os.path.join(PART4_ARTIFACTS_DIR, "qaoa"),
    "comparison": os.path.join(PART4_ARTIFACTS_DIR, "comparison"),
}

# Ensure all directories exist
for d in [PART4_RESULTS_DIR, PART4_ARTIFACTS_DIR] + list(ARTIFACT_DIRS.values()):
    os.makedirs(d, exist_ok=True)

# ── Feature Input ────────────────────────────────────────────────────
# Part 3 produced 16-D latent features from an autoencoder.
# These are the input features for feature selection in Part 4.
INPUT_FEATURE_DIM = 16

# ── Feature Budgets ──────────────────────────────────────────────────
# Different subset sizes to evaluate.
# Must never exceed INPUT_FEATURE_DIM.
FEATURE_BUDGETS = [2, 4, 6, 8]

# ── Classical Feature Selection ──────────────────────────────────────
CLASSICAL_METHODS = ["mutual_info", "rfe"]  # primary pair

# ── QAOA Configuration ───────────────────────────────────────────────
QAOA_NUM_LAYERS = 2       # p parameter — number of QAOA layers
QAOA_SHOTS = 1024         # measurement shots
QAOA_MAX_ITERATIONS = 35  # classical optimiser iterations
QAOA_OPTIMIZER = "Adam"   # classical optimiser name
QAOA_PENALTY_LAMBDA = QFS_LAMBDA  # redundancy penalty weight
QAOA_CARDINALITY_PENALTY = 1.5    # penalty for violating budget
QAOA_IMPORTANCE_WEIGHT = 1.0      # weight for feature relevance

# ── Downstream Evaluation ────────────────────────────────────────────
# The same classifier is used for all feature-subset comparisons.
DOWNSTREAM_CLASSIFIER = "SVM_RBF"
DOWNSTREAM_SEED = RANDOM_SEED

# ── Stability Testing ────────────────────────────────────────────────
QAOA_STABILITY_RUNS = 3
QAOA_STABILITY_SEEDS = [RANDOM_SEED + i for i in range(QAOA_STABILITY_RUNS)]

# ── Reproducibility ──────────────────────────────────────────────────
SEED = RANDOM_SEED
