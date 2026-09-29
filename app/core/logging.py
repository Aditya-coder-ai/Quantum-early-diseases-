"""
app/core/logging.py
Structured logging system with Request ID propagation and PHI scrubbing.
Ensures zero patient clinical values or sensitive secrets are logged to standard output or files.
"""
from __future__ import annotations

import logging
import json
import time
from typing import Any, Dict, Optional

from src.security.audit import scrub_message, SensitiveDataFilter


class StructuredFormatter(logging.Formatter):
    """Formats log records as clean, privacy-scrubbed JSON or key-value entries."""
    def format(self, record: logging.LogRecord) -> str:
        # Scrub message of potential PHI or identifier patterns
        raw_msg = super().format(record)
        clean_msg = scrub_message(raw_msg)
        
        log_entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": clean_msg,
        }
        
        # Attach request_id if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if hasattr(record, "endpoint"):
            log_entry["endpoint"] = record.endpoint
            
        return json.dumps(log_entry)


def setup_app_logging(level: str = "INFO") -> logging.Logger:
    """Configures root application logger with privacy filters."""
    logger = logging.getLogger("mindmatrix_api")
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Avoid duplicate handlers on reload
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
        
    return logger


api_logger = setup_app_logging()
