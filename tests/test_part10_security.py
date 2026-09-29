"""
Automated Test Suite for Part 10: Privacy, Security & Safe Medical Data Handling.

Validates:
1. Patient identifier detection, blocking, stripping, and pseudonymization.
2. Data classification and project data inventory.
3. Secret management, scanner, and .env.example safety.
4. Secure logging, sensitive message scrubbing, and audit logger.
5. Input validation engine (types, ranges, batch limits, safe error responses).
6. Authentication, token HMAC verification, tamper detection, and RBAC permissions.
7. Model artifact policies, SHA-256 integrity, pickle safety, and .gitignore compliance.
8. Live dataset and artifact security compliance.
"""
from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from configs.config import PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR

RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "breast_cancer_raw.csv")
from src.security import (
    # Data Protection
    DataClassification,
    DataInventoryItem,
    DATA_INVENTORY,
    detect_identifier_columns,
    validate_no_identifiers,
    strip_identifiers,
    pseudonymize_identifier,
    pseudonymize_column,
    classify_column,
    classify_dataframe,
    get_data_inventory_report,
    # Secrets
    SECRET_PATTERNS,
    scan_file_for_secrets,
    scan_project_for_secrets,
    get_secret,
    validate_env_example,
    verify_gitignore_excludes_env,
    SecureConfig,
    # Audit & Logging
    scrub_message,
    SensitiveDataFilter,
    get_secure_logger,
    AuditEventType,
    AuditLogger,
    # Input Validation
    WDBC_FEATURE_COUNT,
    WDBC_FEATURE_NAMES,
    MAX_BATCH_SIZE,
    ValidationResult,
    validate_prediction_input,
    safe_error_response,
    # Auth & RBAC
    Role,
    Permission,
    ROLE_PERMISSIONS,
    AuthenticatedUser,
    generate_dev_token,
    validate_token,
    AuthorizationResult,
    check_authorization,
    authenticate_and_authorize,
    # Policies
    ArtifactInfo,
    compute_file_hash,
    inspect_artifact,
    verify_artifact_integrity,
    generate_artifact_manifest,
    check_gitignore_compliance,
    get_data_flow_diagram,
)


# =====================================================================
# 1. Data Protection & Identifier Handling Tests
# =====================================================================
class TestDataProtection:
    def test_identifier_detection(self):
        columns = ["patient_id", "Patient_ID", "Name", "FIRST_NAME", "ssn", "DOB", "mrn", "mean radius"]
        flagged = detect_identifier_columns(columns)
        assert "patient_id" in flagged
        assert "Patient_ID" in flagged
        assert "Name" in flagged
        assert "FIRST_NAME" in flagged
        assert "ssn" in flagged
        assert "DOB" in flagged
        assert "mrn" in flagged
        assert "mean radius" not in flagged

    def test_validate_no_identifiers_raises(self):
        df_dirty = pd.DataFrame({"patient_id": [1, 2], "mean radius": [12.0, 14.5]})
        with pytest.raises(ValueError, match="Identity-related columns detected"):
            validate_no_identifiers(df_dirty)

        df_clean = pd.DataFrame({"mean radius": [12.0, 14.5], "mean texture": [10.1, 11.2]})
        # Should not raise
        validate_no_identifiers(df_clean)

    def test_strip_identifiers(self):
        df = pd.DataFrame({
            "patient_id": [101, 102],
            "first_name": ["Alice", "Bob"],
            "mean radius": [13.54, 15.22],
            "diagnosis": [0, 1]
        })
        clean = strip_identifiers(df)
        assert "patient_id" not in clean.columns
        assert "first_name" not in clean.columns
        assert "mean radius" in clean.columns
        assert "diagnosis" in clean.columns
        # Original df unmodified
        assert "patient_id" in df.columns

    def test_pseudonymization_deterministic_and_oneway(self):
        id_val = "PT_99842"
        salt = "unit-test-salt-abc"
        
        token1 = pseudonymize_identifier(id_val, salt=salt)
        token2 = pseudonymize_identifier(id_val, salt=salt)
        assert token1.startswith("SUBJ_")
        assert token1 == token2  # Deterministic
        assert id_val not in token1  # One-way

        # Different salt produces different token
        token_diff_salt = pseudonymize_identifier(id_val, salt="different-salt")
        assert token1 != token_diff_salt

    def test_pseudonymize_column(self):
        df = pd.DataFrame({
            "patient_id": ["PT1", "PT2", "PT1"],
            "feature": [1.0, 2.0, 3.0]
        })
        res = pseudonymize_column(df, "patient_id", salt="test-salt")
        assert res["patient_id"].iloc[0] == res["patient_id"].iloc[2]
        assert res["patient_id"].iloc[0] != res["patient_id"].iloc[1]
        assert res["patient_id"].iloc[0].startswith("SUBJ_")

    def test_data_classification_and_inventory(self):
        assert classify_column("patient_id") == DataClassification.HIGHLY_SENSITIVE
        assert classify_column("diagnosis") == DataClassification.SENSITIVE
        assert classify_column("mean perimeter") == DataClassification.INTERNAL

        df = pd.DataFrame({"patient_id": [1], "target": [0], "mean radius": [12.0]})
        classified = classify_dataframe(df)
        assert classified["patient_id"] == DataClassification.HIGHLY_SENSITIVE
        assert classified["target"] == DataClassification.SENSITIVE
        assert classified["mean radius"] == DataClassification.INTERNAL

        report = get_data_inventory_report()
        assert "# Data Inventory" in report
        assert "breast_cancer_raw.csv" in report


# =====================================================================
# 2. Secret Management & Secure Configuration Tests
# =====================================================================
class TestSecretsManagement:
    def test_secret_scanner(self, tmp_path):
        bad_file = tmp_path / "bad_code.py"
        bad_file.write_text(
            '# This is a comment\n'
            'api_key = "a1b2c3d4e5f6g7h8i9j0"\n'
            'normal_var = 42\n'
        )
        findings = scan_file_for_secrets(str(bad_file))
        assert len(findings) == 1
        assert findings[0]["line"] == 2
        # Secret value not leaked in finding
        assert "a1b2c3d4e5f6g7h8i9j0" not in str(findings)

    def test_get_secret(self, monkeypatch):
        monkeypatch.setenv("MINDMATRIX_TEST_VAR", "super-secret-value")
        assert get_secret("MINDMATRIX_TEST_VAR") == "super-secret-value"
        assert get_secret("NONEXISTENT_KEY", default="fallback") == "fallback"
        
        with pytest.raises(RuntimeError, match="Required secret 'MISSING_SECRET' is not set"):
            get_secret("MISSING_SECRET", required=True)

    def test_env_example_safety(self):
        env_example_path = os.path.join(PROJECT_ROOT, ".env.example")
        warnings = validate_env_example(env_example_path)
        assert len(warnings) == 0, f"Warnings in .env.example: {warnings}"

    def test_gitignore_excludes_env(self):
        assert verify_gitignore_excludes_env(PROJECT_ROOT) is True

    def test_secure_config_separation(self):
        cfg = SecureConfig(model_name="hybrid_vqc", feature_count=30)
        pub = cfg.get_public_config()
        assert pub["model_name"] == "hybrid_vqc"
        assert pub["feature_count"] == 30
        assert "API_KEY" not in pub
        # Repr redaction
        assert "secrets=[REDACTED]" in repr(cfg)


# =====================================================================
# 3. Secure Logging & Audit Trail Tests
# =====================================================================
class TestAuditLogging:
    def test_scrub_message(self):
        raw = "Error for patient_id: 998877 with api_key = mySecretApiKey123 and password: myPassword!"
        scrubbed = scrub_message(raw)
        assert "998877" not in scrubbed
        assert "mySecretApiKey123" not in scrubbed
        assert "myPassword!" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_sensitive_data_filter(self):
        filt = SensitiveDataFilter()
        import logging
        record = logging.LogRecord("test", logging.INFO, "path", 10, "Accessing patient_id: 12345", (), None)
        filt.filter(record)
        assert "12345" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_audit_logger(self, tmp_path):
        log_file = tmp_path / "test_audit.log"
        logger = AuditLogger(log_path=str(log_file))
        
        entry = logger.log_event(
            event_type=AuditEventType.PREDICTION_REQUEST,
            success=True,
            metadata={"sample_count": 5, "model": "hybrid_vqc"},
            session_id="sess_123",
        )
        assert entry["event_type"] == AuditEventType.PREDICTION_REQUEST
        assert entry["success"] is True
        assert entry["session_id"] == "sess_123"

        events = logger.read_events()
        assert len(events) == 1
        assert events[0]["metadata"]["sample_count"] == 5


# =====================================================================
# 4. Input Validation Engine Tests
# =====================================================================
class TestInputValidation:
    def test_valid_input(self):
        # 1 sample with 30 features
        valid_vec = np.ones((1, WDBC_FEATURE_COUNT))
        res = validate_prediction_input(valid_vec)
        assert res.is_valid is True
        assert len(res.errors) == 0

    def test_null_and_empty_input(self):
        res_none = validate_prediction_input(None)
        assert res_none.is_valid is False
        assert "null/None" in res_none.errors[0]

        res_empty = validate_prediction_input([])
        assert res_empty.is_valid is False

    def test_feature_count_mismatch(self):
        # 20 features instead of 30
        res = validate_prediction_input(np.ones((1, 20)))
        assert res.is_valid is False
        assert any("Feature count mismatch" in e for e in res.errors)

    def test_missing_and_infinite_values(self):
        arr_nan = np.ones((2, WDBC_FEATURE_COUNT))
        arr_nan[0, 5] = np.nan
        res_nan = validate_prediction_input(arr_nan)
        assert res_nan.is_valid is False
        assert any("missing value" in e for e in res_nan.errors)

        arr_inf = np.ones((2, WDBC_FEATURE_COUNT))
        arr_inf[1, 10] = np.inf
        res_inf = validate_prediction_input(arr_inf)
        assert res_inf.is_valid is False
        assert any("infinite value" in e for e in res_inf.errors)

    def test_batch_size_limit(self):
        oversized = np.ones((MAX_BATCH_SIZE + 5, WDBC_FEATURE_COUNT))
        res = validate_prediction_input(oversized)
        assert res.is_valid is False
        assert any("exceeds maximum allowed" in e for e in res.errors)

    def test_safe_error_response(self):
        resp = safe_error_response("Validation failed", status_code=422)
        assert resp["error"] == "Validation failed"
        assert resp["status"] == 422
        assert "Traceback" not in str(resp)


# =====================================================================
# 5. Authentication & Authorization Foundation Tests
# =====================================================================
class TestAuthFoundation:
    def test_generate_and_validate_token(self):
        token = generate_dev_token("user_42", "INFERENCE_USER")
        user = validate_token(token)
        assert user is not None
        assert user.user_id == "user_42"
        assert user.role == Role.INFERENCE_USER

    def test_tampered_token_rejection(self):
        token = generate_dev_token("user_42", "INFERENCE_USER")
        tampered = token[:-4] + "ffff"
        assert validate_token(tampered) is None

        # Altered role in payload
        parts = token.split(":")
        tampered_role = f"{parts[0]}:ADMIN:{parts[2]}"
        assert validate_token(tampered_role) is None

    def test_rbac_least_privilege(self):
        # 1. Inference user
        u_inf = AuthenticatedUser(user_id="doc1", role=Role.INFERENCE_USER)
        assert u_inf.has_permission(Permission.REQUEST_PREDICTION) is True
        assert u_inf.has_permission(Permission.VIEW_MODEL_METADATA) is True
        assert u_inf.has_permission(Permission.RUN_TRAINING) is False
        assert u_inf.has_permission(Permission.MANAGE_MODELS) is False

        # 2. Researcher
        u_res = AuthenticatedUser(user_id="scientist1", role=Role.RESEARCHER)
        assert u_res.has_permission(Permission.REQUEST_PREDICTION) is True
        assert u_res.has_permission(Permission.RUN_TRAINING) is True
        assert u_res.has_permission(Permission.ACCESS_EXPERIMENT_DATA) is True
        assert u_res.has_permission(Permission.MANAGE_USERS) is False

        # 3. Admin
        u_adm = AuthenticatedUser(user_id="root", role=Role.ADMIN)
        for perm in Permission:
            assert u_adm.has_permission(perm) is True

    def test_authenticate_and_authorize_workflow(self):
        token = generate_dev_token("researcher_alice", "RESEARCHER")
        user, authz = authenticate_and_authorize(token, Permission.RUN_TRAINING)
        assert user is not None
        assert authz.allowed is True

        # Try admin permission with researcher token
        user, authz_fail = authenticate_and_authorize(token, Permission.MANAGE_USERS)
        assert user is not None
        assert authz_fail.allowed is False


# =====================================================================
# 6. Model Policies & Artifact Integrity Tests
# =====================================================================
class TestModelPolicies:
    def test_compute_hash_and_verify(self, tmp_path):
        f = tmp_path / "test_file.txt"
        f.write_text("MindMatrix Medical Model Checksum Test")
        h = compute_file_hash(str(f))
        assert len(h) == 64  # SHA-256 hex length
        assert verify_artifact_integrity(str(f), h) is True
        assert verify_artifact_integrity(str(f), "wrong_hash") is False

    def test_inspect_artifact_and_pickle_flag(self, tmp_path):
        joblib_file = tmp_path / "model.joblib"
        joblib_file.write_bytes(b"dummy joblib bytes")
        info = inspect_artifact(str(joblib_file))
        assert info.uses_pickle is True

        json_file = tmp_path / "metrics.json"
        json_file.write_text("{}")
        info_json = inspect_artifact(str(json_file))
        assert info_json.uses_pickle is False

    def test_generate_manifest(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"data-a")
        (tmp_path / "b.bin").write_bytes(b"data-b")
        manifest_out = tmp_path / "manifest.json"
        manifest = generate_artifact_manifest(str(tmp_path), output_path=str(manifest_out))
        assert "a.bin" in manifest
        assert "b.bin" in manifest
        assert os.path.exists(manifest_out)

    def test_gitignore_compliance(self):
        res = check_gitignore_compliance(PROJECT_ROOT)
        assert res["compliant"] is True
        assert len(res["missing_required"]) == 0

    def test_data_flow_diagram(self):
        diag = get_data_flow_diagram()
        assert "# Privacy-Preserving Data Flow" in diag
        assert "Input" in diag
        assert "VQC model" in diag


# =====================================================================
# 7. Live Dataset and Pipeline Integrity Check
# =====================================================================
class TestLiveSystemCompliance:
    def test_raw_data_has_no_identifiers(self):
        if os.path.exists(RAW_DATA_PATH):
            df_raw = pd.read_csv(RAW_DATA_PATH)
            # Must raise no exception
            validate_no_identifiers(df_raw)

    def test_processed_splits_have_no_identifiers(self):
        train_path = os.path.join(DATA_PROCESSED_DIR, "X_train.csv")
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            validate_no_identifiers(df_train)
            assert df_train.shape[1] == WDBC_FEATURE_COUNT
