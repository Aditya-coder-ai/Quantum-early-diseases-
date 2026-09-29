"""
app/core/security.py
Security middleware: API key / token validation, rate limiting, and request size checks.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple, Optional
from fastapi import Request, HTTPException, status, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from src.security.auth import validate_token

# Headers
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_AUTH = HTTPBearer(auto_error=False)


class RateLimiter:
    """Simple in-memory token bucket / sliding window rate limiter by client IP."""
    def __init__(self, requests_per_minute: int = 120):
        self.rpm = requests_per_minute
        self.records: Dict[str, list[float]] = {}

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        now = time.time()
        window_start = now - 60.0
        
        # Clean expired timestamps
        timestamps = self.records.get(client_ip, [])
        valid_timestamps = [t for t in timestamps if t > window_start]
        
        if len(valid_timestamps) >= self.rpm:
            remaining = 0
            self.records[client_ip] = valid_timestamps
            return False, remaining
            
        valid_timestamps.append(now)
        self.records[client_ip] = valid_timestamps
        remaining = self.rpm - len(valid_timestamps)
        return True, remaining


rate_limiter = RateLimiter(requests_per_minute=settings.rate_limit_per_minute)


async def verify_auth_dependency(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(BEARER_AUTH),
) -> None:
    """Verifies API key or JWT Bearer token if authentication is enabled in settings."""
    if not settings.auth_enabled:
        return  # Open in standard development mode

    # Check X-API-Key
    if api_key and settings.api_key and api_key == settings.api_key:
        return

    # Check Bearer JWT token
    if bearer_creds and bearer_creds.credentials:
        user = validate_token(bearer_creds.credentials)
        if user:
            request.state.user = user
            return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid authentication credentials (API key or Bearer token required)"
    )
