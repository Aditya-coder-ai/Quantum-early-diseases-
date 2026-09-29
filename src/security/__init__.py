"""
Part 10: Privacy, Security & Safe Medical Data Handling.

Provides data classification, protection, validation, audit logging,
authentication foundation, and authorization for the MindMatrix
hybrid classical-quantum medical ML pipeline.
"""
from __future__ import annotations

from src.security.data_protection import (
    DataClassification,
    DataInventoryItem,
    DATA_INVENTORY,
    FORBIDDEN_IDENTIFIER_PATTERNS,
    detect_identifier_columns,
    validate_no_identifiers,
    strip_identifiers,
    pseudonymize_identifier,
    pseudonymize_column,
    classify_column,
    classify_dataframe,
    get_data_inventory_report,
)

from src.security.secrets import (
    SECRET_PATTERNS,
    scan_file_for_secrets,
    scan_project_for_secrets,
    get_secret,
    validate_env_example,
    verify_gitignore_excludes_env,
    SecureConfig,
)

from src.security.audit import (
    scrub_message,
    SensitiveDataFilter,
    get_secure_logger,
    AuditEventType,
    AuditLogger,
)

from src.security.validation import (
    WDBC_FEATURE_COUNT,
    WDBC_FEATURE_NAMES,
    MAX_BATCH_SIZE,
    ValidationResult,
    validate_prediction_input,
    safe_error_response,
)

from src.security.auth import (
    Role,
    Permission,
    ROLE_PERMISSIONS,
    AuthenticatedUser,
    generate_dev_token,
    validate_token,
    AuthorizationResult,
    check_authorization,
    authenticate_and_authorize,
)

from src.security.policies import (
    TRUSTED_ARTIFACT_DIRS,
    PICKLE_EXTENSIONS,
    SAFE_EXTENSIONS,
    ArtifactInfo,
    compute_file_hash,
    inspect_artifact,
    verify_artifact_integrity,
    generate_artifact_manifest,
    REQUIRED_GITIGNORE_ENTRIES,
    RECOMMENDED_GITIGNORE_ENTRIES,
    check_gitignore_compliance,
    DATA_FLOW_STAGES,
    get_data_flow_diagram,
)

__all__ = [
    # Data Protection
    "DataClassification",
    "DataInventoryItem",
    "DATA_INVENTORY",
    "FORBIDDEN_IDENTIFIER_PATTERNS",
    "detect_identifier_columns",
    "validate_no_identifiers",
    "strip_identifiers",
    "pseudonymize_identifier",
    "pseudonymize_column",
    "classify_column",
    "classify_dataframe",
    "get_data_inventory_report",
    # Secrets
    "SECRET_PATTERNS",
    "scan_file_for_secrets",
    "scan_project_for_secrets",
    "get_secret",
    "validate_env_example",
    "verify_gitignore_excludes_env",
    "SecureConfig",
    # Audit & Logging
    "scrub_message",
    "SensitiveDataFilter",
    "get_secure_logger",
    "AuditEventType",
    "AuditLogger",
    # Input Validation
    "WDBC_FEATURE_COUNT",
    "WDBC_FEATURE_NAMES",
    "MAX_BATCH_SIZE",
    "ValidationResult",
    "validate_prediction_input",
    "safe_error_response",
    # Auth & Authz
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    "AuthenticatedUser",
    "generate_dev_token",
    "validate_token",
    "AuthorizationResult",
    "check_authorization",
    "authenticate_and_authorize",
    # Policies & Artifact Integrity
    "TRUSTED_ARTIFACT_DIRS",
    "PICKLE_EXTENSIONS",
    "SAFE_EXTENSIONS",
    "ArtifactInfo",
    "compute_file_hash",
    "inspect_artifact",
    "verify_artifact_integrity",
    "generate_artifact_manifest",
    "REQUIRED_GITIGNORE_ENTRIES",
    "RECOMMENDED_GITIGNORE_ENTRIES",
    "check_gitignore_compliance",
    "DATA_FLOW_STAGES",
    "get_data_flow_diagram",
]
