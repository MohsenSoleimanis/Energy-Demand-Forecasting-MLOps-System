"""FastAPI application for the Belgian Energy Demand Forecasting API.

This module is a thin orchestrator: it creates the FastAPI app,
manages the lifespan (creating services and storing them on
``app.state``), and defines endpoints that delegate all business
logic to injected services.

Endpoints:
    POST /predict         - Single prediction
    POST /predict/batch   - Batch predictions (max 1000)
    GET  /health          - Health check with model info
    GET  /metrics         - Prometheus metrics
    GET  /model/info      - Model metadata from MLflow registry
    POST /model/reload    - Hot-reload the production model
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.ml.features.engineering import get_feature_columns, prepare_features
from src.ml.serving.auth import require_api_key
from src.ml.serving.metrics import (
    MODEL_VERSION_GAUGE,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
)
from src.ml.serving.model_service import ModelService
from src.ml.serving.prediction_logger import PredictionLogger
from src.ml.serving.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.shared.config import load_config
from src.shared.s3 import create_s3_client

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).resolve().parents[3] / "configs"


def _load_serving_config() -> dict:
    """Load the serving API configuration file.

    Returns:
        Parsed config dict from ``configs/serving/api.yaml``.
    """
    return load_config(_CONFIGS_DIR / "serving" / "api.yaml")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create services on startup and tear them down on shutdown."""
    from src.shared.config import load_env_file

    load_env_file()

    cfg = _load_serving_config()
    model_name: str = cfg.get("model_name", "energy-demand-forecast")

    # -- ModelService --
    client = mlflow.MlflowClient()
    service = ModelService(model_name=model_name, mlflow_client=client)

    try:
        service.load_production()
        if service.production_version is not None:
            MODEL_VERSION_GAUGE.labels(
                model_name=model_name,
                version=service.production_version.version,
            ).set(1)
    except Exception:
        logger.warning(
            "No production model available. /predict will return 503 "
            "until a model is loaded via POST /model/reload."
        )

    service.load_shadow()

    app.state.model_service = service

    # -- PredictionLogger --
    pl_cfg = cfg.get("prediction_logger", {})
    try:
        pred_logger = PredictionLogger(
            s3_client=create_s3_client(),
            bucket=pl_cfg.get("bucket", "lakehouse"),
            prefix=pl_cfg.get("prefix", "predictions"),
            flush_size=int(pl_cfg.get("flush_size", 50)),
            flush_interval_s=float(pl_cfg.get("flush_interval_s", 60)),
            max_retry_count=int(pl_cfg.get("max_retry_count", 3)),
        )
        app.state.prediction_logger = pred_logger
        logger.info("Prediction logger initialized")
    except Exception as exc:
        logger.warning("Failed to initialize prediction logger: %s", exc)
        app.state.prediction_logger = None

    yield

    # Cleanup
    if app.state.prediction_logger is not None:
        app.state.prediction_logger.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Belgian Energy Demand Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)

try:
    _cfg = _load_serving_config()
    _cors_raw: str = _cfg.get("cors", {}).get(
        "allowed_origins", "http://localhost:3000,http://localhost:8000"
    )
except Exception:
    _cors_raw = "http://localhost:3000,http://localhost:8000"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_raw.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------
def get_model_service(request: Request) -> ModelService:
    """FastAPI dependency that retrieves the ``ModelService`` from app state.

    Returns:
        The application-wide ``ModelService`` instance.

    Raises:
        HTTPException: 503 if no service is available.
    """
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    return service


def get_prediction_logger(request: Request) -> PredictionLogger | None:
    """FastAPI dependency that retrieves the ``PredictionLogger`` from app state.

    Returns:
        The ``PredictionLogger`` instance, or ``None`` if it was not
        initialized.
    """
    return getattr(request.app.state, "prediction_logger", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepare_feature_df(request: PredictionRequest) -> pd.DataFrame:
    """Convert a prediction request into a feature DataFrame.

    Args:
        request: Validated prediction request.

    Returns:
        DataFrame ready for model inference.
    """
    df: pd.DataFrame = pd.DataFrame([request.model_dump()])
    df = prepare_features(df, mode="serving")

    feature_cols: list[str] = get_feature_columns()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df[feature_cols]


def _predict_single(
    request: PredictionRequest,
    service: ModelService,
    pred_logger: PredictionLogger | None,
) -> PredictionResponse:
    """Run prediction for a single request and log the result.

    Args:
        request: Validated prediction request.
        service: The model service.
        pred_logger: Optional prediction logger.

    Returns:
        The prediction response.
    """
    features: pd.DataFrame = _prepare_feature_df(request)

    predicted_load: float = service.predict(features)
    PREDICTION_VALUE.observe(predicted_load)

    response = PredictionResponse(
        timestamp_brussels=request.timestamp_brussels,
        predicted_load_mw=predicted_load,
        model_version=service.production_version.version,
    )

    # Shadow prediction (logged only, never returned)
    shadow_result: dict | None = service.shadow_predict_with_meta(features.copy())

    if pred_logger is not None:
        response_data: dict = response.model_dump()
        if shadow_result is not None:
            response_data.update(shadow_result)
        pred_logger.log(
            request_data=request.model_dump(),
            response_data=response_data,
        )

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    _key: str = Depends(require_api_key),
    service: ModelService = Depends(get_model_service),
    pred_logger: PredictionLogger | None = Depends(get_prediction_logger),
) -> PredictionResponse:
    """Return a single energy demand prediction."""
    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call POST /model/reload first.",
        )
    start: float = time.time()
    try:
        response = _predict_single(request, service, pred_logger)
        PREDICTION_COUNT.labels(endpoint="/predict", status="success").inc()
        return response
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(time.time() - start)


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    _key: str = Depends(require_api_key),
    service: ModelService = Depends(get_model_service),
    pred_logger: PredictionLogger | None = Depends(get_prediction_logger),
) -> list[PredictionResponse]:
    """Return predictions for a batch of up to 1000 requests."""
    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call POST /model/reload first.",
        )
    start: float = time.time()
    try:
        responses = [
            _predict_single(r, service, pred_logger)
            for r in request.predictions
        ]
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="success").inc()
        return responses
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(
            time.time() - start
        )


@app.get("/health", response_model=HealthResponse)
async def health(
    service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    """Return service health and current model information."""
    if service.is_ready:
        return HealthResponse(
            status="healthy",
            model_version=service.production_version.version,
            model_alias="production",
        )
    return HealthResponse(status="degraded")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    _key: str = Depends(require_api_key),
    service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call POST /model/reload first.",
        )
    run = service._client.get_run(service.production_version.run_id)
    return ModelInfoResponse(
        model_name=service.model_name,
        model_version=service.production_version.version,
        metrics=run.data.metrics,
        tags=run.data.tags,
    )


@app.post("/model/reload", response_model=HealthResponse)
async def reload_model(
    _key: str = Depends(require_api_key),
    service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    """Hot-reload the production model from MLflow registry."""
    logger.info("Reloading production model ...")
    service.reload()
    MODEL_VERSION_GAUGE.labels(
        model_name=service.model_name,
        version=service.production_version.version,
    ).set(1)
    logger.info("Model reloaded: version %s", service.production_version.version)
    return HealthResponse(
        status="healthy",
        model_version=service.production_version.version,
        model_alias="production",
    )
