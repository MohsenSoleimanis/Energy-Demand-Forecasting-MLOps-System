"""Pydantic v2 data models for energy demand forecasting.

Defines strict schemas for raw readings, engineered features, train/val/test
splits, and prediction outputs.  All models use ``model_config`` with
``strict=True`` where appropriate so that values are validated eagerly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Raw sensor reading
# ---------------------------------------------------------------------------


class EnergyReading(BaseModel):
    """A single hourly energy meter reading from a building sensor."""

    model_config = ConfigDict(strict=False, arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="UTC timestamp of the reading")
    building_id: str = Field(..., min_length=1, description="Unique building identifier")
    energy_demand_kwh: float = Field(..., gt=0, description="Energy consumption in kWh")
    temperature: float = Field(
        ..., ge=-40.0, le=50.0, description="Outdoor temperature in Celsius"
    )
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity percentage")
    is_holiday: bool = Field(default=False, description="Whether this timestamp falls on a holiday")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        """Accept ISO-format strings as well as datetime objects."""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


# ---------------------------------------------------------------------------
# Engineered feature row
# ---------------------------------------------------------------------------


class FeatureRow(BaseModel):
    """A single row of engineered features ready for model consumption.

    Includes raw fields plus lag features, rolling statistics, and cyclical
    time encodings.  Field-level validation is intentionally relaxed here
    because feature values are produced by ``DataProcessor`` and may take a
    wide range of legitimate values.
    """

    model_config = ConfigDict(strict=False, arbitrary_types_allowed=True)

    # --- original columns ---
    timestamp: datetime
    building_id: str
    energy_demand_kwh: float
    temperature: float
    humidity: float
    is_holiday: bool
    day_of_week: int

    # --- lag features (optional – populated after feature engineering) ---
    lag_1h: float | None = None
    lag_2h: float | None = None
    lag_3h: float | None = None
    lag_6h: float | None = None
    lag_12h: float | None = None
    lag_24h: float | None = None
    lag_48h: float | None = None
    lag_168h: float | None = None

    # --- rolling statistics ---
    rolling_mean_24h: float | None = None
    rolling_std_24h: float | None = None
    rolling_min_24h: float | None = None
    rolling_max_24h: float | None = None
    rolling_mean_168h: float | None = None
    rolling_std_168h: float | None = None
    rolling_min_168h: float | None = None
    rolling_max_168h: float | None = None

    # --- cyclical time encodings ---
    hour_sin: float | None = None
    hour_cos: float | None = None
    day_sin: float | None = None
    day_cos: float | None = None
    month_sin: float | None = None
    month_cos: float | None = None

    # --- derived boolean features ---
    is_weekend: bool | None = None
    is_business_hour: bool | None = None

    # --- degree-day features ---
    heating_degree_days: float | None = None
    cooling_degree_days: float | None = None


# ---------------------------------------------------------------------------
# Train / Validation / Test split container
# ---------------------------------------------------------------------------


class DataSplit(BaseModel):
    """Container holding numpy arrays for a time-series train/val/test split.

    Attributes are stored as generic ``Any`` because Pydantic does not natively
    validate :class:`numpy.ndarray`.  A model-validator ensures that the
    shapes are consistent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X_train: Any = Field(..., description="Training feature matrix")
    y_train: Any = Field(..., description="Training target vector")
    X_val: Any = Field(..., description="Validation feature matrix")
    y_val: Any = Field(..., description="Validation target vector")
    X_test: Any = Field(..., description="Test feature matrix")
    y_test: Any = Field(..., description="Test target vector")
    feature_names: list[str] = Field(default_factory=list, description="Ordered feature names")

    @model_validator(mode="after")
    def _check_shapes(self) -> "DataSplit":
        """Verify that X and y arrays have compatible first-axis lengths."""
        pairs = [
            ("X_train", "y_train"),
            ("X_val", "y_val"),
            ("X_test", "y_test"),
        ]
        for x_name, y_name in pairs:
            x_arr: NDArray[Any] = getattr(self, x_name)
            y_arr: NDArray[Any] = getattr(self, y_name)
            if not isinstance(x_arr, np.ndarray) or not isinstance(y_arr, np.ndarray):
                raise ValueError(f"{x_name} and {y_name} must be numpy arrays")
            if x_arr.shape[0] != y_arr.shape[0]:
                raise ValueError(
                    f"{x_name} rows ({x_arr.shape[0]}) != "
                    f"{y_name} rows ({y_arr.shape[0]})"
                )
        return self

    @property
    def train_size(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def val_size(self) -> int:
        return int(self.X_val.shape[0])

    @property
    def test_size(self) -> int:
        return int(self.X_test.shape[0])


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------


class PredictionResult(BaseModel):
    """Model prediction for a single timestamp and building."""

    model_config = ConfigDict(strict=False)

    timestamp: datetime = Field(..., description="UTC timestamp of the prediction")
    building_id: str = Field(..., min_length=1)
    predicted_kwh: float = Field(..., description="Point prediction in kWh")
    confidence_lower: float = Field(..., description="Lower bound of prediction interval")
    confidence_upper: float = Field(..., description="Upper bound of prediction interval")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    @model_validator(mode="after")
    def _check_interval(self) -> "PredictionResult":
        if self.confidence_lower > self.predicted_kwh:
            raise ValueError(
                "confidence_lower must be <= predicted_kwh "
                f"({self.confidence_lower} > {self.predicted_kwh})"
            )
        if self.confidence_upper < self.predicted_kwh:
            raise ValueError(
                "confidence_upper must be >= predicted_kwh "
                f"({self.confidence_upper} < {self.predicted_kwh})"
            )
        return self


# ---------------------------------------------------------------------------
# Validation result (light dataclass-style model used by DataValidator)
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Outcome of a data-validation check."""

    is_valid: bool = Field(..., description="True if no errors were found")
    errors: list[str] = Field(default_factory=list, description="Hard errors")
    warnings: list[str] = Field(default_factory=list, description="Soft warnings")
    stats: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"ValidationResult({status})"]
        if self.errors:
            parts.append(f"  Errors ({len(self.errors)}):")
            parts.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            parts.append(f"  Warnings ({len(self.warnings)}):")
            parts.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(parts)
