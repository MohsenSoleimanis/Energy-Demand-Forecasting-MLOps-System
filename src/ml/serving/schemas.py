"""
Pydantic v2 request/response schemas for the Energy Demand Forecasting API.

SERVE-002
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uuid


class PredictionRequest(BaseModel):
    """Schema for a single energy demand prediction request."""

    timestamp_brussels: datetime
    temperature_2m: float = Field(ge=-40, le=55)
    relative_humidity_2m: float = Field(ge=0, le=100)
    wind_speed_10m: float = Field(ge=0)
    wind_direction_10m: float = Field(ge=0, le=360)
    shortwave_radiation: float = Field(ge=0)
    precipitation: float = Field(ge=0)
    cloud_cover: float = Field(ge=0, le=100)
    pressure_msl: float = Field(ge=900, le=1100)
    price_eur_mwh: Optional[float] = None
    load_lag_1h: Optional[float] = None
    load_lag_24h: Optional[float] = None
    load_lag_168h: Optional[float] = None


class PredictionResponse(BaseModel):
    """Schema for a single energy demand prediction response."""

    timestamp_brussels: datetime
    predicted_load_mw: float
    model_version: str
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    predicted_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Schema for the health check endpoint response."""

    status: str
    model_version: Optional[str] = None
    model_alias: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Schema for the model info endpoint response."""

    model_name: str
    model_version: str
    metrics: dict
    tags: dict


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests (up to 1000 items)."""

    predictions: list[PredictionRequest] = Field(max_length=1000)
