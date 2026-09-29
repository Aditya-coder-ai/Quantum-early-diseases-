"""
Part 4 — Feature Registry.

Maintains a deterministic mapping from feature indices to names/sources.
The 16-D latent features come from the Part 3 autoencoder (30 → 16 compression).
This registry must be loaded before any feature selection step.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List

from src.feature_selection.config import (
    INPUT_FEATURE_DIM, PART4_ARTIFACTS_DIR
)


@dataclass(frozen=True)
class FeatureEntry:
    """Single feature description."""
    index: int
    name: str
    source: str  # e.g. "autoencoder_latent"
    description: str


def build_latent_feature_registry(n_features: int = INPUT_FEATURE_DIM) -> List[FeatureEntry]:
    """
    Build a deterministic feature registry for Part 3 latent features.

    The autoencoder compresses 30 WDBC clinical features into n_features
    latent dimensions.  Each latent dimension is a learned, non-linear
    combination of the original medical features.
    """
    registry: List[FeatureEntry] = []
    for i in range(n_features):
        entry = FeatureEntry(
            index=i,
            name=f"latent_{i}",
            source="autoencoder_latent_16d",
            description=(
                f"Latent dimension {i} from Part 3 deep autoencoder "
                f"(30→64→32→16 encoder).  Learned, non-linear "
                f"combination of 30 WDBC clinical features."
            ),
        )
        registry.append(entry)
    return registry


def save_registry(registry: List[FeatureEntry], path: str | None = None) -> str:
    """Persist registry to JSON."""
    if path is None:
        path = os.path.join(PART4_ARTIFACTS_DIR, "feature_registry.json")
    data = [asdict(e) for e in registry]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_registry(path: str | None = None) -> List[FeatureEntry]:
    """Load registry from JSON."""
    if path is None:
        path = os.path.join(PART4_ARTIFACTS_DIR, "feature_registry.json")
    with open(path, "r") as f:
        raw = json.load(f)
    return [FeatureEntry(**item) for item in raw]


def get_feature_names(registry: List[FeatureEntry]) -> List[str]:
    """Return ordered feature names."""
    return [e.name for e in sorted(registry, key=lambda e: e.index)]


def get_selected_names(
    registry: List[FeatureEntry], indices: List[int]
) -> List[str]:
    """Return feature names for selected indices."""
    name_map = {e.index: e.name for e in registry}
    return [name_map[i] for i in sorted(indices)]


def validate_registry(registry: List[FeatureEntry], expected_dim: int) -> None:
    """Verify registry integrity."""
    assert len(registry) == expected_dim, (
        f"Registry has {len(registry)} entries but expected {expected_dim}"
    )
    indices = [e.index for e in registry]
    assert sorted(indices) == list(range(expected_dim)), (
        f"Registry indices are not contiguous 0..{expected_dim-1}: {indices}"
    )
    names = [e.name for e in registry]
    assert len(set(names)) == len(names), "Duplicate feature names in registry"
