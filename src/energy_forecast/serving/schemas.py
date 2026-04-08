"""Pydantic v2 request / response models for the prediction API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Single prediction request payload."""

    timestamp: datetime = Field(..., description="Forecast target timestamp (ISO-8601).")
    building_id: str = Field(..., description="Unique building identifier.")
    temperature: float = Field(..., description="Outdoor temperature in °C.")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity (%).")
    features: dict[str, Any] | None = Field(
        default=None,
        description="Optional extra features to include in the feature vector.",
    )


class BatchPredictionRequest(BaseModel):
    """Batch of prediction requests."""

    predictions: list[PredictionRequest] = Field(
        ..., min_length=1, description="List of individual prediction requests."
    )


# ── Responses ─────────────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Single prediction result."""

    building_id: str
    timestamp: datetime
    predicted_kwh: float
    confidence_lower: float
    confidence_upper: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction result."""

    predictions: list[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    model_loaded: bool
    model_version: str | None = None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model."""

    model_name: str
    model_version: str
    model_stage: str
    metrics: dict[str, Any]
    features_used: list[str]


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: str | None = None
