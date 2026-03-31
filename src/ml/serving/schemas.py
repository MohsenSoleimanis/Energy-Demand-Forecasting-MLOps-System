"""Pydantic v2 request/response schemas for the Energy Demand Forecasting API.

Each schema defines strict field constraints for API validation.
All constraints (ranges, limits) reflect physical plausibility of
Belgian energy market and weather data.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Schema for a single energy demand prediction request.

    All weather fields have physically plausible constraints derived from
    Belgian climate norms.  Optional lag/price fields may be ``None`` when
    the caller does not have historical context (the feature pipeline will
    fill defaults).

    Field constraints:
        temperature_2m: -40 to 55 degC (covers all recorded European extremes).
        relative_humidity_2m: 0 to 100 % (physical limits).
        wind_speed_10m: >= 0 m/s (no upper bound -- gusts can be extreme).
        wind_direction_10m: 0 to 360 degrees (compass bearing).
        shortwave_radiation: >= 0 W/m^2.
        precipitation: >= 0 mm/h.
        cloud_cover: 0 to 100 %.
        pressure_msl: 900 to 1100 hPa (plausible sea-level range).
    """

    timestamp_brussels: datetime = Field(
        ...,
        description="Prediction target timestamp in the Europe/Brussels timezone.",
    )
    temperature_2m: float = Field(
        ge=-40, le=55,
        description="Air temperature at 2 m height in degrees Celsius.",
    )
    relative_humidity_2m: float = Field(
        ge=0, le=100,
        description="Relative humidity at 2 m height as a percentage (0-100).",
    )
    wind_speed_10m: float = Field(
        ge=0,
        description="Wind speed at 10 m height in m/s.",
    )
    wind_direction_10m: float = Field(
        ge=0, le=360,
        description="Wind direction at 10 m height in degrees (0-360).",
    )
    shortwave_radiation: float = Field(
        ge=0,
        description="Downward shortwave radiation flux in W/m^2.",
    )
    precipitation: float = Field(
        ge=0,
        description="Precipitation in mm/h.",
    )
    cloud_cover: float = Field(
        ge=0, le=100,
        description="Total cloud cover as a percentage (0-100).",
    )
    pressure_msl: float = Field(
        ge=900, le=1100,
        description="Mean sea level pressure in hPa.",
    )
    price_eur_mwh: float | None = Field(
        default=None,
        description="Day-ahead electricity price in EUR/MWh (optional).",
    )
    load_lag_1h: float | None = Field(
        default=None,
        description="Actual load 1 hour ago in MW (optional).",
    )
    load_lag_24h: float | None = Field(
        default=None,
        description="Actual load 24 hours ago in MW (optional).",
    )
    load_lag_168h: float | None = Field(
        default=None,
        description="Actual load 168 hours (7 days) ago in MW (optional).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timestamp_brussels": "2025-01-15T14:00:00",
                    "temperature_2m": 5.2,
                    "relative_humidity_2m": 78.0,
                    "wind_speed_10m": 4.5,
                    "wind_direction_10m": 220.0,
                    "shortwave_radiation": 150.0,
                    "precipitation": 0.0,
                    "cloud_cover": 60.0,
                    "pressure_msl": 1013.25,
                    "price_eur_mwh": 45.0,
                    "load_lag_24h": 9500.0,
                    "load_lag_168h": 9800.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema for a single energy demand prediction response.

    Every response carries a unique ``prediction_id`` (UUID4) for
    traceability and a ``predicted_at`` wall-clock timestamp so
    downstream consumers can measure serving latency.
    """

    timestamp_brussels: datetime = Field(
        ...,
        description="The target timestamp that was predicted.",
    )
    predicted_load_mw: float = Field(
        ...,
        description="Predicted total load in megawatts.",
    )
    model_version: str = Field(
        ...,
        description="MLflow model version used for this prediction.",
    )
    prediction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this prediction (UUID4).",
    )
    predicted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC wall-clock time when the prediction was made.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timestamp_brussels": "2025-01-15T14:00:00",
                    "predicted_load_mw": 9650.3,
                    "model_version": "5",
                    "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "predicted_at": "2025-01-15T12:30:00",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Schema for the health check endpoint response.

    ``status`` is ``"healthy"`` when a production model is loaded,
    ``"degraded"`` otherwise.
    """

    status: str = Field(
        ...,
        description="Service health status: 'healthy' or 'degraded'.",
    )
    model_version: str | None = Field(
        default=None,
        description="Currently loaded model version, if any.",
    )
    model_alias: str | None = Field(
        default=None,
        description="MLflow alias of the loaded model (e.g. 'production').",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "model_version": "5",
                    "model_alias": "production",
                }
            ]
        }
    }


class ModelInfoResponse(BaseModel):
    """Schema for the model info endpoint response.

    Exposes MLflow run metrics and tags for the loaded model version.
    """

    model_name: str = Field(
        ...,
        description="Registered model name in MLflow.",
    )
    model_version: str = Field(
        ...,
        description="Model version number.",
    )
    metrics: dict = Field(
        ...,
        description="Training/evaluation metrics from the MLflow run.",
    )
    tags: dict = Field(
        ...,
        description="Tags attached to the MLflow run.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_name": "energy-demand-forecast",
                    "model_version": "5",
                    "metrics": {"test_mape": 0.042, "test_rmse": 312.5},
                    "tags": {"mlflow.runName": "lgbm-v5"},
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests.

    Limited to 1000 items per request to bound memory and latency.
    """

    predictions: list[PredictionRequest] = Field(
        ...,
        max_length=1000,
        description="List of prediction requests (max 1000).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predictions": [
                        {
                            "timestamp_brussels": "2025-01-15T14:00:00",
                            "temperature_2m": 5.2,
                            "relative_humidity_2m": 78.0,
                            "wind_speed_10m": 4.5,
                            "wind_direction_10m": 220.0,
                            "shortwave_radiation": 150.0,
                            "precipitation": 0.0,
                            "cloud_cover": 60.0,
                            "pressure_msl": 1013.25,
                        }
                    ]
                }
            ]
        }
    }
