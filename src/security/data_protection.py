"""
Data classification, inventory, minimization, and pseudonymization.

Classifies project data items into PUBLIC / INTERNAL / SENSITIVE /
HIGHLY_SENSITIVE categories.  Provides utilities to strip identifier
columns before they reach the ML pipeline and to generate deterministic
pseudonymous tokens when linkage is required.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ── Classification Levels ────────────────────────────────────────────
class DataClassification(Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    SENSITIVE = "SENSITIVE"
    HIGHLY_SENSITIVE = "HIGHLY_SENSITIVE"


# ── Data Inventory Entry ─────────────────────────────────────────────
@dataclass
class DataInventoryItem:
    name: str
    classification: DataClassification
    storage_location: str
    purpose: str
    retention: str
    access: str


# ── Project Data Inventory ───────────────────────────────────────────
# Built from actual inspection of the repository.
DATA_INVENTORY: List[DataInventoryItem] = [
    # -- Raw Dataset --
    DataInventoryItem(
        name="breast_cancer_raw.csv",
        classification=DataClassification.INTERNAL,
        storage_location="data/raw/",
        purpose="Source dataset for training pipeline (UCI WDBC, no patient identifiers)",
        retention="Duration of research project",
        access="Pipeline scripts, preprocessing module",
    ),
    # -- Processed Splits --
    DataInventoryItem(
        name="X_train.csv / X_val.csv / X_test.csv",
        classification=DataClassification.INTERNAL,
        storage_location="data/processed/",
        purpose="Scaled feature matrices for model training and evaluation",
        retention="Duration of research project",
        access="Training, evaluation, inference modules",
    ),
    DataInventoryItem(
        name="y_train.csv / y_val.csv / y_test.csv",
        classification=DataClassification.INTERNAL,
        storage_location="data/processed/",
        purpose="Binary target labels (0=Malignant, 1=Benign)",
        retention="Duration of research project",
        access="Training, evaluation modules",
    ),
    # -- Model Artifacts --
    DataInventoryItem(
        name="preprocessing_pipeline.joblib / scaler.joblib",
        classification=DataClassification.INTERNAL,
        storage_location="models/",
        purpose="Fitted preprocessing transformers (imputer + scaler)",
        retention="Tied to trained model version",
        access="Inference engine, pipeline module",
    ),
    DataInventoryItem(
        name="autoencoder.pth",
        classification=DataClassification.INTERNAL,
        storage_location="models/",
        purpose="Trained autoencoder for dimensionality reduction (30D → 16D)",
        retention="Tied to trained model version",
        access="Inference engine",
    ),
    DataInventoryItem(
        name="vqc_model.pt / vqc_weights.npy",
        classification=DataClassification.INTERNAL,
        storage_location="models/",
        purpose="Trained VQC model weights for quantum classification",
        retention="Tied to trained model version",
        access="Inference engine",
    ),
    DataInventoryItem(
        name="baseline_*.joblib",
        classification=DataClassification.INTERNAL,
        storage_location="models/",
        purpose="Classical baseline models (LogReg, SVM, RF, MLP)",
        retention="Tied to trained model version",
        access="Comparison module, evaluation",
    ),
    # -- Feature Selection Artifacts --
    DataInventoryItem(
        name="selected_features_quantum.json / selected_features_classical.json",
        classification=DataClassification.INTERNAL,
        storage_location="results/",
        purpose="QAOA and MI feature selection results (feature indices and names)",
        retention="Tied to trained model version",
        access="Inference engine, pipeline module",
    ),
    # -- Experiment Results --
    DataInventoryItem(
        name="Experiment logs and metrics (*.json, *.csv)",
        classification=DataClassification.INTERNAL,
        storage_location="results/ and experiments/",
        purpose="Training metrics, comparison tables, explainability reports",
        retention="Duration of research project",
        access="Researchers, evaluation modules",
    ),
    # -- Configuration --
    DataInventoryItem(
        name="configs/*.py, configs/*.yaml",
        classification=DataClassification.INTERNAL,
        storage_location="configs/",
        purpose="Hyperparameters, paths, pipeline configuration",
        retention="Duration of research project",
        access="All pipeline modules",
    ),
    # -- Metadata --
    DataInventoryItem(
        name="dataset_report.json / feature_schema.json / preprocessing_metadata.json",
        classification=DataClassification.INTERNAL,
        storage_location="data/metadata/",
        purpose="Dataset validation reports and feature schema documentation",
        retention="Duration of research project",
        access="Pipeline modules, documentation",
    ),
    # -- Audit Logs (created by Part 10) --
    DataInventoryItem(
        name="audit.log",
        classification=DataClassification.SENSITIVE,
        storage_location="logs/",
        purpose="Security and access audit trail",
        retention="Minimum 90 days or project duration",
        access="Admin role only",
    ),
]


# ── Identifier Columns to Block from ML Pipeline ────────────────────
# These column name patterns must NEVER enter the feature matrix.
FORBIDDEN_IDENTIFIER_PATTERNS: List[str] = [
    r"(?i)^patient[_\s]?id$",
    r"(?i)^subject[_\s]?id$",
    r"(?i)^name$",
    r"(?i)^first[_\s]?name$",
    r"(?i)^last[_\s]?name$",
    r"(?i)^full[_\s]?name$",
    r"(?i)^phone$",
    r"(?i)^telephone$",
    r"(?i)^email$",
    r"(?i)^e[_\s]?mail$",
    r"(?i)^address$",
    r"(?i)^ssn$",
    r"(?i)^social[_\s]?security",
    r"(?i)^national[_\s]?id",
    r"(?i)^passport",
    r"(?i)^government[_\s]?id",
    r"(?i)^hospital[_\s]?id",
    r"(?i)^mrn$",           # Medical Record Number
    r"(?i)^medical[_\s]?record",
    r"(?i)^date[_\s]?of[_\s]?birth",
    r"(?i)^dob$",
    r"(?i)^zip[_\s]?code$",
    r"(?i)^postal[_\s]?code$",
]

# Compiled for performance
_COMPILED_FORBIDDEN = [re.compile(p) for p in FORBIDDEN_IDENTIFIER_PATTERNS]


def detect_identifier_columns(columns: List[str]) -> List[str]:
    """
    Scan column names for forbidden identifier patterns.

    Returns:
        List of column names that match forbidden identifier patterns.
    """
    flagged = []
    for col in columns:
        for pattern in _COMPILED_FORBIDDEN:
            if pattern.search(col.strip()):
                flagged.append(col)
                break
    return flagged


def validate_no_identifiers(df: pd.DataFrame) -> None:
    """
    Raise ValueError if any identifier columns are present in the DataFrame.

    This is a hard gate: the feature matrix MUST NOT contain identity data.
    """
    flagged = detect_identifier_columns(list(df.columns))
    if flagged:
        raise ValueError(
            f"Identity-related columns detected in feature matrix and MUST be "
            f"removed before model training/inference: {flagged}"
        )


def strip_identifiers(
    df: pd.DataFrame,
    additional_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with all detected identifier columns removed.
    Optionally remove additional specified columns.

    Does NOT modify the original DataFrame.
    """
    cols_to_drop = set(detect_identifier_columns(list(df.columns)))
    if additional_columns:
        cols_to_drop.update(additional_columns)

    existing_drops = [c for c in cols_to_drop if c in df.columns]
    if existing_drops:
        return df.drop(columns=existing_drops)
    return df.copy()


# ── Pseudonymization ────────────────────────────────────────────────
def pseudonymize_identifier(
    identifier: str,
    salt: Optional[str] = None,
) -> str:
    """
    Generate a deterministic, non-reversible pseudonymous token from an identifier.

    Uses HMAC-SHA256 with a secret salt (from environment or provided).
    The original identifier CANNOT be recovered from the token.

    Args:
        identifier: The original identifier string (e.g., patient ID).
        salt: HMAC key.  Falls back to PSEUDONYMIZATION_SALT env var,
              then to a project-specific default (suitable only for
              research prototypes, NOT production).

    Returns:
        Hex-encoded pseudonymous token (first 16 chars of HMAC-SHA256).
    """
    if salt is None:
        salt = os.environ.get(
            "PSEUDONYMIZATION_SALT",
            "mindmatrix-research-prototype-salt-CHANGE-IN-PRODUCTION",
        )
    token = hmac.new(
        salt.encode("utf-8"),
        identifier.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return f"SUBJ_{token}"


def pseudonymize_column(
    df: pd.DataFrame,
    column: str,
    salt: Optional[str] = None,
) -> pd.DataFrame:
    """
    Replace an identifier column with pseudonymous tokens.

    Returns a new DataFrame; the original is not modified.
    """
    if column not in df.columns:
        return df.copy()
    result = df.copy()
    result[column] = result[column].astype(str).apply(
        lambda x: pseudonymize_identifier(x, salt)
    )
    return result


# ── Data Classification Helper ──────────────────────────────────────
def classify_column(column_name: str) -> DataClassification:
    """
    Classify a single column name based on its content type.
    """
    # Identifiers → HIGHLY_SENSITIVE
    if detect_identifier_columns([column_name]):
        return DataClassification.HIGHLY_SENSITIVE

    # Target / diagnosis columns → SENSITIVE
    if re.match(r"(?i)^(target|diagnosis|label|outcome)", column_name):
        return DataClassification.SENSITIVE

    # Numeric clinical features → INTERNAL
    return DataClassification.INTERNAL


def classify_dataframe(df: pd.DataFrame) -> Dict[str, DataClassification]:
    """
    Classify all columns in a DataFrame.

    Returns:
        Dictionary mapping column name → DataClassification.
    """
    return {col: classify_column(col) for col in df.columns}


def get_data_inventory_report() -> str:
    """
    Return the data inventory as a formatted markdown string.
    """
    lines = ["# Data Inventory\n"]
    lines.append("| Data Item | Classification | Location | Purpose | Retention | Access |")
    lines.append("|-----------|---------------|----------|---------|-----------|--------|")
    for item in DATA_INVENTORY:
        lines.append(
            f"| {item.name} | {item.classification.value} | "
            f"`{item.storage_location}` | {item.purpose} | "
            f"{item.retention} | {item.access} |"
        )
    return "\n".join(lines)
