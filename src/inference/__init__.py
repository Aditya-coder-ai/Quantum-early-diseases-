"""
Production inference engine.

Loads serialized pipeline artifacts (preprocessing pipeline, autoencoder,
angle scaler, VQC weights) to perform inference on new patient samples.
Augmentation steps (SMOTE / QGAN) are training-only and never applied here.
"""
from src.inference.predict import MedicalInferenceEngine

__all__ = [
    "MedicalInferenceEngine",
]
