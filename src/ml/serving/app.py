"""
FastAPI application for the Belgian Energy Demand Forecasting API.

SERVE-001

Endpoints:
    POST /predict         - Single prediction
    POST /predict/batch   - Batch predictions (max 1000)
    GET  /health          - Health check with model info
    GET  /metrics         - Prometheus metrics
    GET  /model/info      - Model metadata from MLflow registry
    POST /model/reload    - Hot-reload the production model
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.ml.features.feature_engineering import prepare_features
from src.ml.serving.metrics import (
    MODEL_VERSION_GAUGE,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
)
from src.ml.serving.model_loader import load_production_model
from src.ml.serving.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model state
# ---------------------------------------------------------------------------
_model = None
_model_version = None
_model_name = "energy-demand-forecast"


def _set_model(model, model_version):
    """Update the module-level model and version references."""
    global _model, _model_version
    _model = model
    _model_version = model_version
    if model_version is not None:
        MODEL_VERSION_GAUGE.labels(
            model_name=_model_name,
            version=model_version.version,
        ).set(1)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model on startup."""
    # Ensure MLflow can reach MinIO for artifact storage
    import os
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

    logger.info("Loading production model from MLflow registry ...")
    model, version = load_production_model(_model_name)
    _set_model(model, version)
    if model is None:
        logger.warning(
            "No production model available. /predict will return 503 until "
            "a model is loaded via POST /model/reload."
        )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Belgian Energy Demand Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _predict_single(request: PredictionRequest) -> PredictionResponse:
    """Run prediction for a single request. Assumes _model is not None."""
    from src.ml.features.feature_engineering import get_feature_columns

    df = pd.DataFrame([request.model_dump()])
    df = prepare_features(df, mode="serving")

    # Ensure all feature columns exist and are numeric
    for col in get_feature_columns():
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    prediction = _model.predict(df[get_feature_columns()])
    predicted_load = float(prediction[0])
    PREDICTION_VALUE.observe(predicted_load)
    return PredictionResponse(
        timestamp_brussels=request.timestamp_brussels,
        predicted_load_mw=predicted_load,
        model_version=_model_version.version,
    )


async def _log_prediction_to_minio(response: PredictionResponse) -> None:
    """Log prediction asynchronously (non-blocking, best-effort)."""
    try:
        # Placeholder: in production this would write to MinIO / S3
        logger.debug(
            "Logged prediction %s: %.2f MW",
            response.prediction_id,
            response.predicted_load_mw,
        )
    except Exception:
        logger.exception("Failed to log prediction to MinIO")


def _require_model():
    """Raise 503 if no model is loaded."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call POST /model/reload first.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Return a single energy demand prediction."""
    _require_model()
    start = time.time()
    try:
        response = _predict_single(request)
        PREDICTION_COUNT.labels(endpoint="/predict", status="success").inc()
        # Fire-and-forget async logging
        asyncio.create_task(_log_prediction_to_minio(response))
        return response
    except Exception as e:
        PREDICTION_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(time.time() - start)


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Return predictions for a batch of up to 1000 requests."""
    _require_model()
    start = time.time()
    try:
        responses = [_predict_single(r) for r in request.predictions]
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="success").inc()
        for resp in responses:
            asyncio.create_task(_log_prediction_to_minio(resp))
        return responses
    except Exception as e:
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(
            time.time() - start
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service health and current model information."""
    if _model is not None:
        return HealthResponse(
            status="healthy",
            model_version=_model_version.version,
            model_alias="production",
        )
    return HealthResponse(status="degraded")


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return metadata about the currently loaded model."""
    _require_model()
    import mlflow

    client = mlflow.MlflowClient()
    run = client.get_run(_model_version.run_id)
    return ModelInfoResponse(
        model_name=_model_name,
        model_version=_model_version.version,
        metrics=run.data.metrics,
        tags=run.data.tags,
    )


@app.post("/model/reload", response_model=HealthResponse)
async def reload_model():
    """Hot-reload the production model from MLflow registry."""
    logger.info("Reloading production model ...")
    model, version = load_production_model(_model_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Failed to reload model from MLflow registry.",
        )
    _set_model(model, version)
    logger.info(f"Model reloaded: version {version.version}")
    return HealthResponse(
        status="healthy",
        model_version=version.version,
        model_alias="production",
    )
