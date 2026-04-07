"""API route definitions for the energy-demand prediction service."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from energy_forecast.monitoring.prometheus_metrics import track_prediction_metrics
from energy_forecast.serving.dependencies import ModelManager, get_model_manager
from energy_forecast.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Application start time (set once at import) ──────────────────────────────
_START_TIME: float = time.time()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_feature_vector(req: PredictionRequest) -> np.ndarray:
    """Convert a *PredictionRequest* into a numeric feature vector.

    The ordering is intentionally deterministic so it matches the column order
    expected by the trained model:
        [temperature, humidity, hour_sin, hour_cos, ...]
    """
    ts: datetime = req.timestamp
    hour = ts.hour + ts.minute / 60.0
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)

    base: list[float] = [req.temperature, req.humidity, hour_sin, hour_cos]

    if req.features:
        for _key in sorted(req.features):
            val = req.features[_key]
            if isinstance(val, (int, float)):
                base.append(float(val))

    return np.array(base, dtype=np.float64).reshape(1, -1)


def _predict_single(
    model: Any, req: PredictionRequest, model_version: str
) -> PredictionResponse:
    """Run a single prediction and build the response."""
    features = _build_feature_vector(req)
    start = time.perf_counter()

    prediction: np.ndarray = np.asarray(model.predict(features)).ravel()
    latency = time.perf_counter() - start

    predicted_kwh = float(prediction[0])

    # Compute confidence interval: ±10 % of predicted value (min ±5 kWh)
    margin = max(abs(predicted_kwh) * 0.10, 5.0)
    confidence_lower = predicted_kwh - margin
    confidence_upper = predicted_kwh + margin

    track_prediction_metrics(
        latency=latency,
        value=predicted_kwh,
        model_version=model_version,
        status="success",
    )

    return PredictionResponse(
        building_id=req.building_id,
        timestamp=req.timestamp,
        predicted_kwh=round(predicted_kwh, 4),
        confidence_lower=round(confidence_lower, 4),
        confidence_upper=round(confidence_upper, 4),
        model_version=model_version,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Single prediction",
)
def predict(
    request: PredictionRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> PredictionResponse:
    """Return a single energy-demand prediction for the given building and timestamp."""
    try:
        model = manager.get_model()
        info = manager.get_model_info()
        return _predict_single(model, request, info.get("model_version", "unknown"))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        track_prediction_metrics(latency=0, value=0, model_version="unknown", status="error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Batch prediction",
)
def predict_batch(
    request: BatchPredictionRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> BatchPredictionResponse:
    """Return predictions for a batch of requests."""
    start = time.perf_counter()
    try:
        model = manager.get_model()
        info = manager.get_model_info()
        model_version = info.get("model_version", "unknown")

        results: list[PredictionResponse] = []
        for req in request.predictions:
            results.append(_predict_single(model, req, model_version))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return BatchPredictionResponse(
            predictions=results,
            processing_time_ms=round(elapsed_ms, 2),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health(
    manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    """Report service health and model readiness."""
    info = manager.get_model_info()
    uptime = time.time() - _START_TIME
    return HealthResponse(
        status="healthy" if manager.is_loaded else "degraded",
        model_loaded=manager.is_loaded,
        model_version=info.get("model_version"),
        uptime_seconds=round(uptime, 2),
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Model information",
)
def model_info(
    manager: ModelManager = Depends(get_model_manager),
) -> ModelInfoResponse:
    """Return metadata about the currently active model."""
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="No model is currently loaded.")

    info = manager.get_model_info()
    return ModelInfoResponse(
        model_name=info.get("model_name", "unknown"),
        model_version=info.get("model_version", "unknown"),
        model_stage=info.get("model_stage", "unknown"),
        metrics=info.get("metrics", {}),
        features_used=info.get("features_used", []),
    )
