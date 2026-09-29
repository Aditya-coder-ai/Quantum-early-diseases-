"""
Part 8: Feature Lineage & Multi-Level Mapping Tracker.
Tracks feature transformations across pipeline tiers:
Raw 30D Features -> 16D Autoencoder Latent Space -> 8D QAOA Selected Features.
Explicitly contrasts QAOA Optimization Objective with SHAP Model Attribution.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional, Tuple, Union

from configs.config import PROJECT_ROOT


class FeatureLineageTracker:
    """
    Manages multi-tier feature tracing and transparently contrasts
    QAOA feature selection relevance against SHAP prediction attribution.
    """
    def __init__(
        self,
        raw_feature_names: List[str],
        selected_indices: List[int],
        latent_dim: int = 16,
        qaoa_artifact_path: Optional[str] = None,
        autoencoder_model: Optional[torch.nn.Module] = None,
    ):
        self.raw_feature_names = list(raw_feature_names)
        self.selected_indices = list(selected_indices)
        self.latent_dim = latent_dim
        self.k = len(selected_indices)
        self.selected_names = [f"latent_{i}" for i in self.selected_indices]
        self.autoencoder_model = autoencoder_model

        # Load QAOA metadata if available
        self.qaoa_meta = {}
        if qaoa_artifact_path and os.path.exists(qaoa_artifact_path):
            with open(qaoa_artifact_path, "r") as f:
                self.qaoa_meta = json.load(f)
        else:
            default_qaoa = os.path.join(PROJECT_ROOT, "features", "selected", "qaoa", "qaoa_selection_primary.json")
            if os.path.exists(default_qaoa):
                with open(default_qaoa, "r") as f:
                    self.qaoa_meta = json.load(f)

    def get_selection_summary(self) -> pd.DataFrame:
        """
        Summarizes the status of each latent representation dimension under QAOA selection.
        """
        rows = []
        for i in range(self.latent_dim):
            is_sel = i in self.selected_indices
            rank = self.selected_indices.index(i) + 1 if is_sel else None
            rows.append({
                "latent_index": i,
                "feature_name": f"latent_{i}",
                "selected_by_qaoa": is_sel,
                "selection_rank": rank,
                "status": "SELECTED" if is_sel else "EXCLUDED",
            })
        return pd.DataFrame(rows)

    def trace_raw_feature_importance_to_latent(self) -> pd.DataFrame:
        """
        Approximates the influence of each 30D raw clinical feature onto the
        compressed latent dimensions using the absolute weight chain of the encoder.
        W_total = |W_1| @ |W_2| @ |W_3|  shape (30, 16)
        """
        if self.autoencoder_model is None:
            # Fallback uniform proxy if model weights are not loaded
            return pd.DataFrame(
                np.ones((len(self.raw_feature_names), self.latent_dim)) / len(self.raw_feature_names),
                index=self.raw_feature_names,
                columns=[f"latent_{i}" for i in range(self.latent_dim)]
            )

        with torch.no_grad():
            encoder = self.autoencoder_model.encoder
            # Extract Linear layer weights
            linear_weights = [
                layer.weight.abs().cpu().numpy()
                for layer in encoder
                if isinstance(layer, torch.nn.Linear)
            ]

        # Chain multiplication of linear layers: (30 -> 64) @ (64 -> 32) @ (32 -> 16)
        # linear_weights: [ (64, 30), (32, 64), (16, 32) ]
        if len(linear_weights) >= 3:
            w1 = linear_weights[0].T  # (30, 64)
            w2 = linear_weights[1].T  # (64, 32)
            w3 = linear_weights[2].T  # (32, 16)
            total_influence = w1 @ w2 @ w3  # (30, 16)
        else:
            total_influence = np.ones((len(self.raw_feature_names), self.latent_dim))

        # Normalize column-wise so each latent dimension sums to 1.0
        col_sums = total_influence.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        norm_influence = total_influence / col_sums

        return pd.DataFrame(
            norm_influence,
            index=self.raw_feature_names,
            columns=[f"latent_{i}" for i in range(self.latent_dim)]
        )

    def build_qaoa_vs_shap_comparison(
        self,
        shap_importance: Dict[str, float],
        classical_importance: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Explicitly compares QAOA objective selection against SHAP model-attributed importance.
        
        Args:
            shap_importance: Mapping {feature_name: mean_abs_shap}
            classical_importance: Optional mapping {feature_name: baseline_importance}
        """
        rows = []
        # Sort SHAP importance to compute ranks
        sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
        shap_ranks = {name: rank + 1 for rank, (name, _) in enumerate(sorted_shap)}

        for feat_name in self.selected_names:
            idx = int(feat_name.split("_")[-1])
            is_qaoa = idx in self.selected_indices
            qaoa_rank = self.selected_indices.index(idx) + 1 if is_qaoa else None
            shap_val = shap_importance.get(feat_name, 0.0)
            shap_rank = shap_ranks.get(feat_name, None)
            class_val = classical_importance.get(feat_name, 0.0) if classical_importance else None

            rows.append({
                "feature": feat_name,
                "selected_by_qaoa": is_qaoa,
                "qaoa_selection_rank": qaoa_rank,
                "shap_importance": round(float(shap_val), 5),
                "shap_rank": shap_rank,
                "classical_importance": round(float(class_val), 5) if class_val is not None else None,
                "alignment_note": (
                    "High SHAP & QAOA Priority" if (shap_rank and shap_rank <= 3 and qaoa_rank and qaoa_rank <= 3)
                    else "Divergent Attribution" if (shap_rank and abs((shap_rank or 0) - (qaoa_rank or 0)) >= 4)
                    else "Moderate Alignment"
                )
            })

        df = pd.DataFrame(rows).sort_values("shap_rank")
        return df
