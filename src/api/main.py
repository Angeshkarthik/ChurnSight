"""FastAPI Application Entry Point.

Configures application lifespan, CORS middleware, safe error handlers, and route registration.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    get_cors_origins,
)
from src.api.routes import router
from src.api.service import get_model_service

# Configure logger
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to safely initialize ML model artifacts on startup."""
    logger.info("Initializing FastAPI Customer Intelligence service on startup...")
    # Safe model pre-loading
    try:
        service = get_model_service()
        if service.is_ready:
            logger.info("Champion model ready: %s (threshold=%.2f)", service.model_name, service.operational_threshold)
        else:
            logger.warning("Application started in degraded mode: model failed to load.")
    except Exception as exc:
        logger.error("Failed to load model during startup lifespan: %s", exc)
    yield
    logger.info("Shutting down FastAPI Customer Intelligence service...")


def create_application() -> FastAPI:
    """Factory function creating and configuring the FastAPI application."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS explicitly for local development (no unrestricted wildcard origins, credentials disabled)
    allowed_origins = get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Safe validation error handler (preserves structured feedback without leaking internals)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(item) for item in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            err_type = err.get("type", "validation_error")
            errors.append({
                "field": loc,
                "message": msg,
                "type": err_type,
            })
        logger.info("Validation failed on %s %s: %s", request.method, request.url.path, errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": errors,
                "message": "Input validation failed. Please check feature names, types, and allowed values.",
            },
        )

    # Safe catch-all internal server error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled internal exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred while processing the request.",
            },
        )

    # Register API routes
    app.include_router(router)

    return app


# Export singleton app instance
app = create_application()
