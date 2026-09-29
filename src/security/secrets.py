"""
Secret management and secure configuration.

Ensures that secrets (API keys, passwords, tokens) are loaded ONLY from
environment variables or a .env file — never hardcoded in source code,
YAML, JSON, or notebooks.

Provides a safe configuration wrapper that separates public config
(hyperparameters, paths) from secret config (credentials).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Secret Patterns for Scanning ─────────────────────────────────────
SECRET_PATTERNS = [
    re.compile(r"""(?i)(?:api[_\-\s]?key|apikey)\s*[:=]\s*['"]([A-Za-z0-9_\-]{16,})['"]"""),
    re.compile(r"""(?i)(?:secret[_\-\s]?key|secretkey)\s*[:=]\s*['"]([A-Za-z0-9_\-]{16,})['"]"""),
    re.compile(r"""(?i)(?:password|passwd|pwd)\s*[:=]\s*['"](.{8,})['"]"""),
    re.compile(r"""(?i)(?:token|auth_token|access_token)\s*[:=]\s*['"]([A-Za-z0-9_\-\.]{16,})['"]"""),
    re.compile(r"""(?i)(?:private[_\-\s]?key)\s*[:=]\s*['"](.+)['"]"""),
    re.compile(r"""(?i)(?:database[_\-\s]?url|db[_\-\s]?url)\s*[:=]\s*['"]((?:postgres|mysql|mongodb).+)['"]"""),
    re.compile(r"""(?i)(?:aws[_\-\s]?secret)\s*[:=]\s*['"]([A-Za-z0-9/+=]{30,})['"]"""),
]


def scan_file_for_secrets(filepath: str) -> List[Dict[str, Any]]:
    """
    Scan a single file for potential hardcoded secrets.

    Returns a list of findings (line number + pattern matched).
    The actual secret value is NOT included in the finding.
    """
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                for pattern in SECRET_PATTERNS:
                    if pattern.search(line):
                        findings.append({
                            "file": filepath,
                            "line": line_num,
                            "pattern": pattern.pattern[:40] + "...",
                            "risk": "Potential hardcoded secret detected",
                        })
                        break  # one finding per line is sufficient
    except (OSError, UnicodeDecodeError):
        pass  # skip unreadable files
    return findings


def scan_project_for_secrets(
    project_root: str,
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Walk the project tree and scan source files for hardcoded secrets.

    Args:
        project_root: Root directory of the project.
        extensions: File extensions to scan (default: .py, .yaml, .yml, .json, .cfg, .ini).
        exclude_dirs: Directory names to skip (default: .git, __pycache__, .venv, node_modules).

    Returns:
        List of findings. Secret values are NEVER included.
    """
    if extensions is None:
        extensions = [".py", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml"]
    if exclude_dirs is None:
        exclude_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", "tests", "test"}

    all_findings = []
    for root, dirs, files in os.walk(project_root):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in files:
            if any(fname.endswith(ext) for ext in extensions):
                fpath = os.path.join(root, fname)
                all_findings.extend(scan_file_for_secrets(fpath))
    return all_findings


# ── Secure Environment Variable Access ──────────────────────────────
def get_secret(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Retrieve a secret value from environment variables.

    Args:
        key: Environment variable name.
        default: Fallback if not set and not required.
        required: If True, raise RuntimeError when the variable is missing.

    Returns:
        The secret string value, or default.
    """
    value = os.environ.get(key)
    if value is None:
        if required:
            raise RuntimeError(
                f"Required secret '{key}' is not set. "
                f"Set it as an environment variable or in a .env file (never in source code)."
            )
        return default
    return value


# ── .env Validation ─────────────────────────────────────────────────
# Tokens that mark a value as an intentional placeholder rather than a
# real credential. Without these, a correct placeholder such as
# "your-auth-secret-here" is indistinguishable from a real secret.
_PLACEHOLDER_TOKENS = (
    "your", "placeholder", "changeme", "change_me", "change-me",
    "example", "sample", "dummy", "redacted", "todo", "insert",
    "replace", "specify", "notreal", "fake",
)
_WRAPPED_PLACEHOLDER = re.compile(r"^(?:<[^>]+>|\$\{[^}]+\})$")


def _is_placeholder(value: str) -> bool:
    """
    Return True if a value is recognizably a placeholder, not a real secret.
    """
    if not value:
        return True
    if _WRAPPED_PLACEHOLDER.match(value):
        return True
    lowered = value.lower()
    if any(token in lowered for token in _PLACEHOLDER_TOKENS):
        return True
    return value.startswith("...") or value.endswith("...")


def validate_env_example(env_example_path: str) -> List[str]:
    """
    Verify that a .env.example file contains no real secret values.

    Placeholder values (e.g. "your-auth-secret-here", "<token>") are
    accepted. Returns list of warnings. Empty list = safe.
    """
    warnings = []
    if not os.path.exists(env_example_path):
        return ["No .env.example file found."]

    with open(env_example_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Check format: KEY=value
            if "=" in stripped:
                key, _, value = stripped.partition("=")
                value = value.strip().strip("'\"")
                # Flag if value looks like an actual secret (long alphanumeric)
                if (
                    len(value) > 15
                    and not _is_placeholder(value)
                    and re.match(r"^[A-Za-z0-9_\-/+=]+$", value)
                ):
                    warnings.append(
                        f"Line {line_num}: Value for '{key.strip()}' looks like a real secret. "
                        f"Use a placeholder like 'your-{key.strip().lower()}-here' instead."
                    )
    return warnings


def verify_gitignore_excludes_env(project_root: str) -> bool:
    """
    Check that .env is listed in .gitignore.
    """
    gitignore_path = os.path.join(project_root, ".gitignore")
    if not os.path.exists(gitignore_path):
        return False

    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped in (".env", ".env*", "*.env"):
                return True
    return False


# ── Safe Configuration Wrapper ──────────────────────────────────────
@dataclass
class SecureConfig:
    """
    Separates safe (public) configuration from secret configuration.

    Public values (hyperparameters, paths, seeds) are stored directly.
    Secret values are read from environment variables at access time.
    """
    # Public configuration (safe to log, store, commit)
    model_name: str = "hybrid_vqc"
    feature_count: int = 30
    latent_dim: int = 16
    selected_dim: int = 8
    num_qubits: int = 8
    random_seed: int = 42
    environment: str = "development"

    # Secret keys — these are ENV VAR NAMES, not the actual values
    _secret_keys: List[str] = field(default_factory=lambda: [
        "API_KEY",
        "SECRET_KEY",
        "DATABASE_URL",
        "PSEUDONYMIZATION_SALT",
    ])

    def get_secret(self, key: str, required: bool = False) -> Optional[str]:
        """Access a secret value from environment (never stored in memory longer than needed)."""
        return get_secret(key, required=required)

    def get_public_config(self) -> Dict[str, Any]:
        """Return only the public/safe configuration values (safe to log)."""
        return {
            "model_name": self.model_name,
            "feature_count": self.feature_count,
            "latent_dim": self.latent_dim,
            "selected_dim": self.selected_dim,
            "num_qubits": self.num_qubits,
            "random_seed": self.random_seed,
            "environment": self.environment,
        }

    def __repr__(self) -> str:
        """Safe repr that never exposes secret values."""
        return (
            f"SecureConfig(model_name={self.model_name!r}, "
            f"environment={self.environment!r}, "
            f"feature_count={self.feature_count}, "
            f"secrets=[REDACTED])"
        )
