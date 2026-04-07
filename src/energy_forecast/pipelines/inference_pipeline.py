"""End-to-end inference pipeline: load model -> validate input -> predict -> monitor."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from energy_forecast.data.validator import DataValidator
from energy_forecast.features.engineering import FeatureEngineer
from energy_forecast.models.base import BaseForecaster
from energy_forecast.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of an inference pipeline run."""
    predictions: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    processing_time_ms: float
    input_shape: tuple[int, ...]


class InferencePipeline:
    """Orchestrates the inference workflow."""

    def __init__(self, model: BaseForecaster, feature_names: list[str],
                 scaler: Optional[Any] = None, residuals: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        # Store residuals from training for confidence intervals
        self._residual_std = float(np.std(residuals)) if residuals is not None else None

    def predict(self, df: pd.DataFrame) -> InferenceResult:
        """Run the full inference pipeline."""
        start_time = time.time()

        # Step 1: Feature engineering
        engineer = FeatureEngineer()
        df = engineer.create_time_features(df)
        if "energy_demand_kwh" in df.columns:
            df = engineer.create_lag_features(df, "energy_demand_kwh", [1, 2, 3, 6, 12, 24])
            df = engineer.create_rolling_features(df, "energy_demand_kwh", [6, 12, 24])
        df = engineer.create_weather_features(df)
        df = engineer.create_calendar_features(df)

        # Step 2: Select features
        available_features = [f for f in self.feature_names if f in df.columns]
        X = df[available_features].fillna(0).values

        # Step 3: Normalize if scaler available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Step 4: Predict
        predictions = self.model.predict(X)

        # Step 5: Confidence intervals
        if self._residual_std is not None:
            margin = 1.96 * self._residual_std  # 95% CI
            confidence_lower = predictions - margin
            confidence_upper = predictions + margin
        else:
            confidence_lower = predictions * 0.9
            confidence_upper = predictions * 1.1

        processing_time_ms = (time.time() - start_time) * 1000

        return InferenceResult(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            processing_time_ms=processing_time_ms,
            input_shape=X.shape,
        )

    def predict_single(self, features: dict[str, float]) -> dict[str, float]:
        """Predict for a single input (for API serving)."""
        df = pd.DataFrame([features])
        result = self.predict(df)
        return {
            "predicted_kwh": float(result.predictions[0]),
            "confidence_lower": float(result.confidence_lower[0]),
            "confidence_upper": float(result.confidence_upper[0]),
            "processing_time_ms": result.processing_time_ms,
        }
