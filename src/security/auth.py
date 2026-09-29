"""
Authentication and authorization foundation.

Since the current MindMatrix project does NOT have an HTTP API server,
this module provides the **interface and architecture** for authentication
and role-based access control.  The actual API integration will be
completed when the API layer is built (Part 11+).

Current state:
- Token-based authentication helpers (validate, generate for dev).
- Role-based authorization with least-privilege model.
- Ready to be wired into any WSGI/ASGI framework (Flask, FastAPI, etc.).

Security notes:
- Development tokens are HMAC-SHA256 based.
- Production deployments MUST use a proper identity provider.
- This module does NOT implement a full enterprise IAM system.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ── Roles ────────────────────────────────────────────────────────────
class Role(Enum):
    ADMIN = "ADMIN"
    RESEARCHER = "RESEARCHER"
    INFERENCE_USER = "INFERENCE_USER"


# ── Permissions ──────────────────────────────────────────────────────
class Permission(Enum):
    # Inference
    REQUEST_PREDICTION = "REQUEST_PREDICTION"
    VIEW_MODEL_METADATA = "VIEW_MODEL_METADATA"

    # Research
    ACCESS_EXPERIMENT_DATA = "ACCESS_EXPERIMENT_DATA"
    ACCESS_TRAINING_METRICS = "ACCESS_TRAINING_METRICS"
    RUN_TRAINING = "RUN_TRAINING"

    # Admin
    MANAGE_MODELS = "MANAGE_MODELS"
    MANAGE_CONFIG = "MANAGE_CONFIG"
    VIEW_AUDIT_LOGS = "VIEW_AUDIT_LOGS"
    MANAGE_USERS = "MANAGE_USERS"


# ── Role → Permission Mapping (Least Privilege) ─────────────────────
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.INFERENCE_USER: {
        Permission.REQUEST_PREDICTION,
        Permission.VIEW_MODEL_METADATA,
    },
    Role.RESEARCHER: {
        Permission.REQUEST_PREDICTION,
        Permission.VIEW_MODEL_METADATA,
        Permission.ACCESS_EXPERIMENT_DATA,
        Permission.ACCESS_TRAINING_METRICS,
        Permission.RUN_TRAINING,
    },
    Role.ADMIN: set(Permission),  # Admin gets all permissions
}


@dataclass
class AuthenticatedUser:
    """Represents a validated user/principal."""
    user_id: str
    role: Role
    authenticated_at: float = field(default_factory=time.time)

    @property
    def permissions(self) -> Set[Permission]:
        return ROLE_PERMISSIONS.get(self.role, set())

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

    def __repr__(self) -> str:
        return f"AuthenticatedUser(user_id={self.user_id!r}, role={self.role.value})"


# ── Token Authentication ─────────────────────────────────────────────
def _get_auth_secret() -> str:
    """
    Get the authentication secret from environment.
    Falls back to a development-only default.
    """
    secret = os.environ.get("MINDMATRIX_AUTH_SECRET")
    if secret is None:
        secret = "mindmatrix-dev-auth-secret-CHANGE-IN-PRODUCTION"
    return secret


def generate_dev_token(user_id: str, role: str) -> str:
    """
    Generate a development authentication token.

    Format: user_id:role:signature

    WARNING: This is for development/testing only.
    Production MUST use a proper token system (JWT, OAuth2, etc.).
    """
    secret = _get_auth_secret()
    payload = f"{user_id}:{role}"
    signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]
    return f"{payload}:{signature}"


def validate_token(token: str) -> Optional[AuthenticatedUser]:
    """
    Validate an authentication token and return the authenticated user.

    Returns None if the token is invalid or malformed.
    """
    if not token or not isinstance(token, str):
        return None

    parts = token.split(":")
    if len(parts) != 3:
        return None

    user_id, role_str, provided_sig = parts

    # Validate role
    try:
        role = Role(role_str.upper())
    except ValueError:
        return None

    # Verify signature
    secret = _get_auth_secret()
    payload = f"{user_id}:{role_str}"
    expected_sig = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]

    if not hmac.compare_digest(provided_sig, expected_sig):
        return None

    return AuthenticatedUser(user_id=user_id, role=role)


# ── Authorization Check ──────────────────────────────────────────────
@dataclass
class AuthorizationResult:
    """Result of an authorization check."""
    allowed: bool
    reason: str


def check_authorization(
    user: Optional[AuthenticatedUser],
    required_permission: Permission,
) -> AuthorizationResult:
    """
    Check if a user has the required permission.

    Args:
        user: The authenticated user (or None if unauthenticated).
        required_permission: The permission required for the operation.

    Returns:
        AuthorizationResult indicating whether access is allowed.
    """
    if user is None:
        return AuthorizationResult(
            allowed=False,
            reason="Authentication required.",
        )

    if user.has_permission(required_permission):
        return AuthorizationResult(
            allowed=True,
            reason=f"User '{user.user_id}' with role '{user.role.value}' is authorized.",
        )

    return AuthorizationResult(
        allowed=False,
        reason=(
            f"User '{user.user_id}' with role '{user.role.value}' "
            f"does not have permission '{required_permission.value}'."
        ),
    )


# ── Convenience Functions ────────────────────────────────────────────
def authenticate_and_authorize(
    token: str,
    required_permission: Permission,
) -> tuple[Optional[AuthenticatedUser], AuthorizationResult]:
    """
    Combined authentication + authorization check.

    Returns:
        (user, auth_result) — user is None if authentication failed.
    """
    user = validate_token(token)
    if user is None:
        return None, AuthorizationResult(
            allowed=False,
            reason="Invalid or missing authentication token.",
        )

    result = check_authorization(user, required_permission)
    return user, result
