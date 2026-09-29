"""
Secure logging and audit trail.

Provides:
1. A safe logger that scrubs sensitive patterns before writing.
2. An audit logger that records security-relevant events (dataset access,
   model load, prediction request, auth failure, etc.) with timestamps
   and non-sensitive metadata — never raw medical data.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import PROJECT_ROOT


# ── Log directory ────────────────────────────────────────────────────
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


# ── Sensitive Patterns to Scrub ──────────────────────────────────────
_SCRUB_PATTERNS = [
    (re.compile(r"(?i)(patient[_\s]?id\s*[:=]\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(name\s*[:=]\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_\s]?key\s*[:=]\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(password\s*[:=]\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(token\s*[:=]\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(secret[_\s]?key\s*[:=]\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(ssn\s*[:=]\s*)(\S+)"), r"\1[REDACTED]"),
]


def scrub_message(message: str) -> str:
    """
    Redact sensitive patterns from a log message.
    """
    result = message
    for pattern, replacement in _SCRUB_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ── Safe Logging Filter ──────────────────────────────────────────────
class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that scrubs sensitive data from log records
    before they are written to any handler.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = scrub_message(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: scrub_message(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    scrub_message(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


def get_secure_logger(
    name: str = "mindmatrix",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger with the sensitive-data filter attached.

    Messages containing patient IDs, API keys, passwords, etc.
    will be automatically redacted before being written.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(SensitiveDataFilter())
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        fh.addFilter(SensitiveDataFilter())
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ── Audit Event Types ────────────────────────────────────────────────
class AuditEventType:
    DATASET_ACCESS = "DATASET_ACCESS"
    MODEL_LOADED = "MODEL_LOADED"
    PREDICTION_REQUEST = "PREDICTION_REQUEST"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    AUTHZ_FAILURE = "AUTHZ_FAILURE"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    SECRET_ACCESS = "SECRET_ACCESS"
    SECURITY_SCAN = "SECURITY_SCAN"
    DATA_EXPORT = "DATA_EXPORT"


# ── Audit Logger ─────────────────────────────────────────────────────
class AuditLogger:
    """
    Structured audit logger for security-relevant events.

    Writes JSON-lines to a dedicated audit log file.
    Each entry contains timestamp, event type, success/failure,
    and non-sensitive metadata.  Raw medical data is NEVER logged.
    """

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            log_path = os.path.join(LOGS_DIR, "audit.log")
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log_event(
        self,
        event_type: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record an audit event.

        Args:
            event_type: One of AuditEventType constants.
            success: Whether the operation succeeded.
            metadata: Non-sensitive contextual data (e.g., sample count,
                      model name, endpoint).  Must NOT contain medical data.
            session_id: Optional request/session identifier.

        Returns:
            The audit entry dict (for testing).
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "success": success,
            "session_id": session_id,
            "metadata": metadata or {},
        }

        # Safety: scrub any accidentally included sensitive data
        entry_str = json.dumps(entry, default=str)
        entry_str = scrub_message(entry_str)

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry_str + "\n")
        except OSError:
            pass  # Fail open for audit (log, don't crash)

        return json.loads(entry_str)

    def read_events(self, last_n: Optional[int] = None) -> list:
        """
        Read audit events from the log file.

        Args:
            last_n: If specified, return only the last N events.
        """
        if not os.path.exists(self.log_path):
            return []
        events = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        events.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
        if last_n is not None:
            return events[-last_n:]
        return events
