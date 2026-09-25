"""
Classical feature extraction orchestrator for Part 3.

Executes dimensionality reduction across multiple target dimensions (4, 8, 12, 16)
using PCA and Neural Autoencoders. Validates and saves representation artifacts.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, MODELS_DIR, RESULTS_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT
)
from src.features.reduction import PCAReducer, AutoencoderReducer
from src.features.validation import FeatureQualityValidator


def run_feature_extraction_pipeline(
    target_dims: list[int] = [4, 8, 12, 16],
    save_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Run complete feature extraction experiment across specified target dimensions.
    """
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "features", "classical")
    
    models_out_dir = os.path.join(MODELS_DIR, "feature_extractor")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_out_dir, exist_ok=True)

    # Load preprocessed splits
    X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))

    input_dim = X_train.shape[1]
    validator = FeatureQualityValidator()
    extraction_results = {}

    print("\n" + "=" * 60)
    print("PART 3: CLASSICAL FEATURE EXTRACTION & DIMENSIONALITY REDUCTION")
    print("=" * 60)
    print(f"Original feature space: {input_dim} features (Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)})")
    print(f"Target dimensions: {target_dims}")

    for dim in target_dims:
        print(f"\n--- Dimension {dim} ---")
        
        # 1. PCA Reduction
        pca_reducer = PCAReducer(n_components=dim, random_state=RANDOM_SEED)
        X_tr_pca = pca_reducer.fit_transform(X_train)
        X_va_pca = pca_reducer.transform(X_val)
        X_te_pca = pca_reducer.transform(X_test)

        pca_model_path = os.path.join(models_out_dir, f"pca_{dim}.joblib")
        pca_reducer.save(pca_model_path)

        pca_meta = validator.validate_features(X_tr_pca, X_va_pca, X_te_pca, method_name="PCA", dim=dim)
        pca_meta["explained_variance_ratio"] = [float(v) for v in pca_reducer.explained_variance_ratio_]
        pca_meta["total_explained_variance"] = float(pca_reducer.total_explained_variance_)

        pca_files = validator.export_feature_set(
            X_tr_pca, X_va_pca, X_te_pca, output_dir=save_dir, prefix=f"pca_{dim}"
        )

        with open(os.path.join(save_dir, f"metadata_pca_{dim}.json"), "w") as f:
            json.dump(pca_meta, f, indent=2)

        print(f"  [PCA {dim}D] Total explained variance: {pca_meta['total_explained_variance']*100:.2f}% | Valid: {pca_meta['is_valid']}")

        # 2. Autoencoder Reduction
        ae_reducer = AutoencoderReducer(
            latent_dim=dim, input_dim=input_dim, random_seed=RANDOM_SEED
        )
        ae_reducer.fit(X_train, X_val)
        X_tr_ae = ae_reducer.transform(X_train)
        X_va_ae = ae_reducer.transform(X_val)
        X_te_ae = ae_reducer.transform(X_test)

        ae_model_path = os.path.join(models_out_dir, f"autoencoder_{dim}.pth")
        ae_reducer.save(ae_model_path)

        ae_meta = validator.validate_features(X_tr_ae, X_va_ae, X_te_ae, method_name="Autoencoder", dim=dim)
        ae_meta["best_epoch"] = ae_reducer.history.get("best_epoch", 0)
        ae_meta["best_val_loss"] = ae_reducer.history.get("best_val_loss", 0.0)

        ae_files = validator.export_feature_set(
            X_tr_ae, X_va_ae, X_te_ae, output_dir=save_dir, prefix=f"autoencoder_{dim}"
        )

        with open(os.path.join(save_dir, f"metadata_autoencoder_{dim}.json"), "w") as f:
            json.dump(ae_meta, f, indent=2)

        print(f"  [AE {dim}D] Best val loss: {ae_meta['best_val_loss']:.6f} @ epoch {ae_meta['best_epoch']+1} | Valid: {ae_meta['is_valid']}")

        extraction_results[f"pca_{dim}"] = {
            "meta": pca_meta,
            "files": pca_files,
            "model_path": pca_model_path,
        }
        extraction_results[f"autoencoder_{dim}"] = {
            "meta": ae_meta,
            "files": ae_files,
            "model_path": ae_model_path,
        }

    summary_path = os.path.join(save_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "input_dim": input_dim,
            "target_dims": target_dims,
            "results_keys": list(extraction_results.keys()),
        }, f, indent=2)

    print(f"\n[FEATURES] Feature extraction complete. Artifacts saved in {save_dir}")
    return extraction_results


if __name__ == "__main__":
    run_feature_extraction_pipeline()
