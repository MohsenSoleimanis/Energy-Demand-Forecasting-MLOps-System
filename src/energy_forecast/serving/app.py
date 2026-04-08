"""FastAPI application factory for the energy-demand prediction service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from energy_forecast.serving.dependencies import get_config, get_model_manager
from energy_forecast.serving.middleware import RequestLoggingMiddleware
from energy_forecast.serving.routes import router
from energy_forecast.serving.schemas import ErrorResponse

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and return a fully-configured :class:`FastAPI` application.

    The returned app:

    * Loads the prediction model during startup via a lifespan context manager.
    * Exposes prediction, health, and model-info routes.
    * Includes request-logging middleware (with Prometheus metrics).
    * Enables CORS for the origins listed in ``serving_config.yaml``.
    * Registers global exception handlers for common error types.
    """
    config = get_config()
    api_cfg = config.get("api", {})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup / shutdown lifecycle hook."""
        manager = get_model_manager()
        model_name = api_cfg.get("model_name", "energy-demand-forecaster")
        model_stage = api_cfg.get("model_stage", "Production")
        logger.info(
            "Loading model %s (stage=%s) on startup …", model_name, model_stage
        )
        manager.load_model(model_name=model_name, stage=model_stage)
        yield
        logger.info("Application shutting down.")

    app = FastAPI(
        title="Energy Demand Forecasting API",
        version="1.0.0",
        description=(
            "Production prediction service for building-level energy demand. "
            "Backed by models trained and registered through the MLOps pipeline."
        ),
        lifespan=lifespan,
    )

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Middleware (order matters – outermost first) ───────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    cors_origins: list[str] = api_cfg.get(
        "cors_origins",
        ["http://localhost:3000", "http://localhost:8080"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        body = ErrorResponse(error="Validation error", detail=str(exc))
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        body = ErrorResponse(error="Internal error", detail=str(exc))
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception")
        body = ErrorResponse(error="Unexpected error", detail=str(exc))
        return JSONResponse(status_code=500, content=body.model_dump())

    return app
