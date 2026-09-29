"""
app/config.py
Central Configuration for FastAPI Inference Service.
Reads from environment variables with safe, reproducible defaults.
"""
from __future__ import annotations

import os
from typing import List, Optional
from pydantic import BaseModel, Field

from configs.config import PROJECT_ROOT


def resolve_default_artifact_dir() -> str:
    """Finds the primary or most recently generated experiment artifact directory."""
    # Preferred stable directory
    preferred = os.path.join(PROJECT_ROOT, "artifacts", "experiments", "config_b_qaoa_vqc_orig_fa7a21e45376")
    if os.path.exists(preferred):
        return preferred

    # Fallback to searching artifacts/experiments
    base_dir = os.path.join(PROJECT_ROOT, "artifacts", "experiments")
    if os.path.exists(base_dir):
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if subdirs:
            subdirs.sort(key=os.path.getmtime, reverse=True)
            return subdirs[0]
            
    # Fallback to models root
    return os.path.join(PROJECT_ROOT, "models")


class Settings(BaseModel):
    """Application runtime settings."""
    app_name: str = "MIndMatrix Hybrid Quantum Disease Detection API"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    
    # Model Artifacts
    artifact_dir: str = Field(default_factory=lambda: os.getenv("MODEL_ARTIFACT_PATH", resolve_default_artifact_dir()))
    model_type: str = Field(default_factory=lambda: os.getenv("MODEL_TYPE", "hybrid_vqc"))
    quantum_backend: str = Field(default_factory=lambda: os.getenv("QUANTUM_BACKEND", "PennyLane default.qubit (Simulation)"))
    
    # Inference parameters
    max_batch_size: int = Field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "100")))
    default_threshold: float = Field(default_factory=lambda: float(os.getenv("CLASSIFICATION_THRESHOLD", "0.5")))
    inference_timeout_s: float = Field(default_factory=lambda: float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "30.0")))
    
    # Security & CORS
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"
            ).split(",")
            if origin.strip()
        ]
    )
    auth_enabled: bool = Field(default_factory=lambda: os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1"))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("API_KEY", None))
    rate_limit_per_minute: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")))


settings = Settings()
