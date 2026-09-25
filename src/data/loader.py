"""
Dataset loader for the Wisconsin Breast Cancer (Diagnostic) dataset.

Loads from scikit-learn, saves raw CSV for traceability, and provides
structured access to features and targets.
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import DATA_RAW_DIR, RANDOM_SEED


def load_dataset(save_raw: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Wisconsin Breast Cancer Diagnostic dataset.
    
    Returns:
        X: DataFrame of shape (569, 30) with named features
        y: Series with binary target (0 = malignant, 1 = benign)
    """
    data = load_breast_cancer()
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    if save_raw:
        raw_path = os.path.join(DATA_RAW_DIR, "breast_cancer_raw.csv")
        df = X.copy()
        df["target"] = y
        df.to_csv(raw_path, index=False)
        print(f"[DATA] Raw dataset saved to {raw_path}")
    
    return X, y


def get_dataset_info(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute comprehensive dataset statistics for the dataset card.
    """
    class_counts = y.value_counts().to_dict()
    class_names = {0: "malignant", 1: "benign"}
    
    info = {
        "name": "Wisconsin Breast Cancer Diagnostic (WDBC)",
        "source": "UCI ML Repository via sklearn.datasets.load_breast_cancer",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "target_variable": "target (0=malignant, 1=benign)",
        "class_distribution": {
            class_names.get(k, k): v for k, v in class_counts.items()
        },
        "imbalance_ratio": round(max(class_counts.values()) / min(class_counts.values()), 3),
        "missing_values": int(X.isnull().sum().sum()),
        "duplicated_rows": int(X.duplicated().sum()),
        "feature_types": "all continuous (float64)",
        "categorical_variables": 0,
        "feature_dtypes": X.dtypes.value_counts().to_dict(),
        "numeric_stats": {
            "min_value": float(X.min().min()),
            "max_value": float(X.max().max()),
            "mean_range": (float(X.mean().min()), float(X.mean().max())),
        },
    }
    return info


def generate_dataset_card(output_path: str | None = None) -> str:
    """Generate and write DATASET_CARD.md from live dataset statistics."""
    X, y = load_dataset(save_raw=True)
    info = get_dataset_info(X, y)
    
    if output_path is None:
        from configs.config import PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, "DATASET_CARD.md")
        
    card_content = f"""# Dataset Card: Wisconsin Breast Cancer (Diagnostic)

## 1. Dataset Overview
- **Dataset Name:** {info['name']}
- **Source:** {info['source']}
- **Task:** Binary Disease Classification
- **Sample Count:** {info['n_samples']} instances
- **Feature Count:** {info['n_features']} continuous medical features
- **Target Variable:** {info['target_variable']}

## 2. Class Distribution
- **Malignant (Class 0):** {info['class_distribution']['malignant']} ({info['class_distribution']['malignant']/info['n_samples']*100:.1f}%)
- **Benign (Class 1):** {info['class_distribution']['benign']} ({info['class_distribution']['benign']/info['n_samples']*100:.1f}%)
- **Imbalance Ratio:** {info['imbalance_ratio']} (Benign : Malignant)

## 3. Data Integrity & Types
- **Missing Values:** {info['missing_values']}
- **Duplicated Rows:** {info['duplicated_rows']}
- **Feature Data Types:** {info['feature_types']}
- **Value Ranges:** Min {info['numeric_stats']['min_value']:.4f} to Max {info['numeric_stats']['max_value']:.4f}

## 4. Clinical Features
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass:
- radius, texture, perimeter, area, smoothness
- compactness, concavity, concave points, symmetry, fractal dimension
(Each measured across mean, standard error, and worst/largest values).

## 5. Usage & Research Safety
This dataset is used solely for developing and benchmarking an experimental hybrid classical-quantum machine learning prototype. Not approved for direct clinical diagnostics.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"[DATA] Dataset card generated at {output_path}")
    return card_content


if __name__ == "__main__":
    X, y = load_dataset()
    info = get_dataset_info(X, y)
    
    print(f"\n{'='*60}")
    print(f"DATASET: {info['name']}")
    print(f"{'='*60}")
    print(f"Samples:          {info['n_samples']}")
    print(f"Features:         {info['n_features']}")
    print(f"Missing values:   {info['missing_values']}")
    print(f"Duplicated rows:  {info['duplicated_rows']}")
    print(f"Class distribution: {info['class_distribution']}")
    print(f"Imbalance ratio:  {info['imbalance_ratio']}")
    print(f"Feature types:    {info['feature_types']}")
