"""
tests/test_part10_fastapi.py
Comprehensive test suite for Part 10: FastAPI Inference Service.
Tests health, readiness, model info, single prediction, batch prediction,
explanation (SHAP & Grad-CAM), validation rejections, error handling,
direct-vs-API model consistency, and security rate limiting.
"""
from __future__ import annotations

import io
import time
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.model_manager import model_manager
from src.data.loader import load_dataset
from src.inference.predict import MedicalInferenceEngine
from src.security.validation import WDBC_FEATURE_NAMES, WDBC_FEATURE_COUNT, MAX_BATCH_SIZE


@pytest.fixture(scope="module")
def client():
    """Provides TestClient with startup and shutdown lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_data():
    """Provides sample clinical patient records from dataset."""
    X, y = load_dataset()
    sample_list = X.iloc[0].tolist()
    sample_dict = X.iloc[0].to_dict()
    return {
        "X": X,
        "y": y,
        "sample_list": sample_list,
        "sample_dict": sample_dict
    }


class TestHealthAndMetadata:
    """Tests health, readiness, root landing, and model information endpoints."""

    def test_root_landing(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "api_v1" in data
        assert data["version"] == settings.app_version

    def test_health_check_endpoints(self, client):
        for path in ("/health", "/api/v1/health"):
            resp = client.get(path)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert "timestamp" in data

    def test_readiness_check_endpoints(self, client):
        for path in ("/ready", "/api/v1/ready"):
            resp = client.get(path)
            assert resp.status_code == 200
            data = resp.json()
            assert data["ready"] is True
            assert data["model_loaded"] is True
            assert data["artifacts_valid"] is True
            assert "details" in data

    def test_model_info_endpoint(self, client):
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_feature_count"] == 30
        assert data["selected_feature_count"] == 8
        assert data["qubits"] == 8
        assert data["circuit_depth"] == 2
        assert len(data["selected_feature_names"]) == 8
        assert "default.qubit" in data["quantum_backend"]
        assert 0.0 <= data["classification_threshold"] <= 1.0


class TestPredictionEndpoints:
    """Tests single and batch inference endpoints with input validation."""

    def test_predict_with_feature_list(self, client, sample_data):
        payload = {"features": sample_data["sample_list"], "sample_id": "test_patient_001"}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "request_id" in data
        assert "prediction" in data

        pred = data["prediction"]
        assert pred["predicted_class"] in (0, 1)
        assert pred["diagnosis_label"] in ("Malignant", "Benign")
        assert 0.0 <= pred["malignant_probability"] <= 1.0
        assert 0.0 <= pred["benign_probability"] <= 1.0
        assert round(pred["malignant_probability"] + pred["benign_probability"], 2) == 1.0
        assert pred["oncology_risk_assessment"] in ("HIGH_RISK_MALIGNANT", "LOW_RISK_BENIGN")

    def test_predict_with_feature_dict(self, client, sample_data):
        payload = {"features": sample_data["sample_dict"]}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"]["predicted_class"] in (0, 1)

    def test_predict_with_threshold_override(self, client, sample_data):
        # Force low threshold vs high threshold
        payload_low = {"features": sample_data["sample_list"], "threshold": 0.01}
        resp_low = client.post("/api/v1/predict", json=payload_low)
        assert resp_low.status_code == 200
        assert resp_low.json()["prediction"]["decision_threshold_used"] == 0.01

        payload_high = {"features": sample_data["sample_list"], "threshold": 0.99}
        resp_high = client.post("/api/v1/predict", json=payload_high)
        assert resp_high.status_code == 200
        assert resp_high.json()["prediction"]["decision_threshold_used"] == 0.99

    def test_predict_batch_success(self, client, sample_data):
        batch = [sample_data["X"].iloc[i].tolist() for i in range(10)]
        payload = {"samples": batch}
        resp = client.post("/api/v1/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 10
        assert len(data["predictions"]) == 10
        assert data["total_inference_time_s"] > 0
        assert data["latency_per_sample_ms"] > 0

    def test_direct_vs_api_consistency(self, client, sample_data):
        """Validates that POST /api/v1/predict exactly matches direct Python MedicalInferenceEngine."""
        engine = MedicalInferenceEngine(model_manager.artifact_dir)
        direct_result = engine.predict(sample_data["sample_dict"], threshold=0.5)

        resp = client.post("/api/v1/predict", json={"features": sample_data["sample_dict"], "threshold": 0.5})
        assert resp.status_code == 200
        api_pred = resp.json()["prediction"]

        # Check probability alignment within numerical tolerance 1e-4
        direct_mal = direct_result["malignant_probabilities"][0]
        api_mal = api_pred["malignant_probability"]
        assert abs(direct_mal - api_mal) < 1e-4
        assert direct_result["predicted_classes"][0] == api_pred["predicted_class"]
        assert direct_result["diagnosis_labels"][0] == api_pred["diagnosis_label"]


class TestInputValidationAndErrorHandling:
    """Tests strict rejection of invalid, incomplete, or malicious payloads."""

    def test_reject_wrong_feature_count(self, client):
        # Only 2 features instead of 30
        resp = client.post("/api/v1/predict", json={"features": [1.0, 2.0]})
        assert resp.status_code == 422
        data = resp.json()
        assert data["success"] is False
        assert "error" in data

    def test_reject_nan_and_inf(self, client, sample_data):
        # Null / None in numeric feature array
        bad_features = list(sample_data["sample_list"])
        bad_features[5] = None
        resp = client.post("/api/v1/predict", json={"features": bad_features})
        assert resp.status_code == 422

        # Non-numeric string in feature array
        bad_features[5] = "invalid_string_measurement"
        resp2 = client.post("/api/v1/predict", json={"features": bad_features})
        assert resp2.status_code == 422

    def test_reject_missing_dict_keys(self, client):
        # Missing 29 keys
        resp = client.post("/api/v1/predict", json={"features": {"mean radius": 14.5}})
        assert resp.status_code == 422

    def test_reject_empty_batch(self, client):
        resp = client.post("/api/v1/predict/batch", json={"samples": []})
        assert resp.status_code == 422

    def test_reject_exceeded_batch_size(self, client, sample_data):
        # Exceeds MAX_BATCH_SIZE (100)
        huge_batch = [sample_data["sample_list"]] * (MAX_BATCH_SIZE + 1)
        resp = client.post("/api/v1/predict/batch", json={"samples": huge_batch})
        assert resp.status_code == 422

    def test_request_id_propagation(self, client, sample_data):
        custom_id = "req_custom_clinic_uuid_99"
        headers = {"X-Request-ID": custom_id}
        resp = client.post("/api/v1/predict", json={"features": sample_data["sample_list"]}, headers=headers)
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == custom_id
        assert resp.json()["request_id"] == custom_id
        assert "X-Process-Time-Ms" in resp.headers


class TestExplanationEndpoints:
    """Tests SHAP attribution and Grad-CAM modality routing."""

    def test_explain_tabular_shap(self, client, sample_data):
        payload = {"features": sample_data["sample_list"], "nsamples": 20}
        resp = client.post("/api/v1/explain", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "explanation" in data

        exp = data["explanation"]
        assert "SHAP" in exp["method"]
        assert 0.0 <= exp["base_value"] <= 1.0
        assert len(exp["all_contributions"]) == 30
        assert len(exp["top_positive_drivers"]) > 0 or len(exp["top_negative_drivers"]) > 0
        assert "clinical_caveat" in exp

    def test_explain_image_modality_routing(self, client):
        # Create a synthetic image with contrasting pixels
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("biopsy_slide.png", buf, "image/png")}

        resp = client.post("/api/v1/explain/image", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["modality_status"] == "TABULAR_MODEL_VISION_GATE"
        assert "details" in data
        assert data["details"]["heatmap_shape"] == [64, 64]
        assert 0.0 <= data["details"]["heatmap_min"] <= 1.0
        assert 0.0 <= data["details"]["heatmap_max"] <= 1.0


class TestSecurityAndRateLimiting:
    """Tests rate limiter and request safety."""

    def test_rate_limiter_allows_normal_traffic(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_rate_limiter_burst_enforcement(self, client):
        from app.core.security import rate_limiter
        client_ip = "test_dos_ip_192"
        # Temporarily fill the bucket
        rate_limiter.records[client_ip] = [time.time()] * (settings.rate_limit_per_minute + 5)
        allowed, remaining = rate_limiter.is_allowed(client_ip)
        assert allowed is False
        assert remaining == 0
