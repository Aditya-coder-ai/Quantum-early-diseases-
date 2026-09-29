"""
app/main.py
Main FastAPI Application Entrypoint.
Initializes lifespan lifecycle, CORS, Request-ID tracing, Rate-Limiting,
global error handlers, and mounts versioned API routers.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.logging import api_logger
from app.core.errors import (
    BaseAppException, app_exception_handler,
    validation_exception_handler, generic_exception_handler
)
from app.core.security import rate_limiter
from app.services.model_manager import model_manager
from app.api.health import router as health_router
from app.api.model import router as model_router
from app.api.prediction import router as prediction_router
from app.api.explanation import router as explanation_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    Loads and validates model artifacts once at startup; cleans up on shutdown.
    """
    api_logger.info("Initializing MIndMatrix Diagnostic Inference Application...")
    try:
        model_manager.load_artifacts(settings.artifact_dir)
        api_logger.info("ModelManager successfully ready for inference traffic.")
    except Exception as e:
        api_logger.error(f"Startup artifact load failed: {e}")
        # In strict production mode, fail fast; in testing/dev, allow app to boot so readiness checks report status 503
        if settings.environment == "production":
            raise e

    yield

    api_logger.info("Shutting down MIndMatrix Inference Service...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-grade REST API service for hybrid classical-quantum medical diagnostic inference. "
        "Operates on 30 continuous cytologic features from fine needle aspirate (FNA) biopsied breast masses."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"]
)


# ── Correlation ID & Rate Limiting Middleware ──
@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    t0 = time.perf_counter()

    # 1. Resolve or generate unique X-Request-ID
    req_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
    request.state.request_id = req_id

    # 2. Check rate limit
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining = rate_limiter.is_allowed(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "request_id": req_id,
                "error": {
                    "type": "RateLimitExceeded",
                    "message": f"Rate limit of {settings.rate_limit_per_minute} req/min exceeded.",
                    "status_code": 429
                }
            },
            headers={"Retry-After": "60", "X-Request-ID": req_id}
        )

    # 3. Process request
    response = await call_next(request)

    # 4. Attach response tracking headers
    process_time_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

    # 5. Safe structured audit logging (never logs patient feature values)
    api_logger.info(
        f"Handled {request.method} {request.url.path} -> Status {response.status_code} "
        f"({process_time_ms:.1f}ms)",
        extra={
            "request_id": req_id,
            "endpoint": request.url.path,
            "latency_ms": round(process_time_ms, 2)
        }
    )

    return response


# ── Exception Handlers ──
app.add_exception_handler(BaseAppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# ── Mount Top-Level Convenience Endpoints ──
app.include_router(health_router)

# ── Mount Versioned API Endpoints (/api/v1) ──
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(model_router, prefix=settings.api_v1_prefix)
app.include_router(prediction_router, prefix=settings.api_v1_prefix)
app.include_router(explanation_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["Root"])
def root_summary():
    """Service landing page and OpenAPI navigation."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "documentation": "/docs",
        "redoc": "/redoc",
        "api_v1": {
            "health": f"{settings.api_v1_prefix}/health",
            "ready": f"{settings.api_v1_prefix}/ready",
            "model_info": f"{settings.api_v1_prefix}/model/info",
            "predict": f"{settings.api_v1_prefix}/predict",
            "predict_batch": f"{settings.api_v1_prefix}/predict/batch",
            "explain": f"{settings.api_v1_prefix}/explain",
            "explain_image": f"{settings.api_v1_prefix}/explain/image",
        }
    }
