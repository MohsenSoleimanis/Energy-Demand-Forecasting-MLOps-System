"""End-to-end training pipeline: data generation -> validation -> features -> training -> evaluation -> registration."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from energy_forecast.data.synthetic import SyntheticDataGenerator
from energy_forecast.data.processor import DataProcessor
from energy_forecast.data.validator import DataValidator
from energy_forecast.features.engineering import FeatureEngineer
from energy_forecast.models.base import BaseForecaster
from energy_forecast.models.linear import LinearForecaster
from energy_forecast.models.xgboost_model import XGBoostForecaster
from energy_forecast.evaluation.metrics import MetricsCalculator
from energy_forecast.evaluation.evaluator import ModelEvaluator
from energy_forecast.training.trainer import Trainer
from energy_forecast.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a training pipeline run."""
    model: BaseForecaster
    metrics: dict[str, float]
    val_metrics: dict[str, float]
    run_id: Optional[str]
    model_version: Optional[str]
    data_shape: tuple[int, ...]
    feature_names: list[str]


class TrainingPipeline:
    """Orchestrates the full training workflow."""

    MODEL_REGISTRY = {
        "linear": LinearForecaster,
        "xgboost": XGBoostForecaster,
    }

    def __init__(self, data_config: Optional[dict] = None,
                 model_config: Optional[dict] = None,
                 model_type: str = "xgboost"):
        self.data_config = data_config or self._load_default_config("data_config")
        self.model_config = model_config or self._load_default_config("model_config")
        self.model_type = model_type

    @staticmethod
    def _load_default_config(name: str) -> dict:
        try:
            return load_config(name)
        except FileNotFoundError:
            return {}

    def run(self, df: Optional[pd.DataFrame] = None) -> PipelineResult:
        """Execute the full training pipeline."""
        # Step 1: Generate or use provided data
        logger.info("Step 1: Preparing data")
        if df is None:
            synthetic_config = self.data_config.get("synthetic", {})
            generator = SyntheticDataGenerator(
                num_buildings=synthetic_config.get("num_buildings", 5),
                start_date=synthetic_config.get("start_date", "2022-01-01"),
                end_date=synthetic_config.get("end_date", "2023-06-30"),
                random_seed=synthetic_config.get("random_seed", 42),
            )
            df = generator.generate()
        logger.info(f"Data shape: {df.shape}")

        # Step 2: Validate raw data
        logger.info("Step 2: Validating raw data")
        validator = DataValidator()
        validation = validator.validate_raw_data(df)
        if not validation.is_valid:
            logger.warning(f"Data validation warnings: {validation.errors}")

        # Step 3: Feature engineering
        logger.info("Step 3: Engineering features")
        engineer = FeatureEngineer()
        processing_config = self.data_config.get("processing", {})
        lags = processing_config.get("lag_features", [1, 2, 3, 6, 12, 24])
        windows = processing_config.get("rolling_windows", [6, 12, 24])

        df = engineer.create_time_features(df)
        df = engineer.create_lag_features(df, "energy_demand_kwh", lags)
        df = engineer.create_rolling_features(df, "energy_demand_kwh", windows)
        df = engineer.create_weather_features(df)
        df = engineer.create_calendar_features(df)

        # Drop rows with NaN from lag/rolling features
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Features shape after engineering: {df.shape}")

        # Step 4: Prepare train/val/test splits
        logger.info("Step 4: Splitting data")
        processor = DataProcessor(processing_config)
        feature_names = engineer.get_feature_names(df)
        target_col = processing_config.get("target_column", "energy_demand_kwh")

        X = df[feature_names].values
        y = df[target_col].values

        train_ratio = processing_config.get("train_ratio", 0.7)
        val_ratio = processing_config.get("val_ratio", 0.15)

        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # Normalize
        X_train, X_val, X_test, scaler = processor.normalize(X_train, X_val, X_test)

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Step 5: Create and train model
        logger.info(f"Step 5: Training {self.model_type} model")
        model_params = self.model_config.get("models", {}).get(self.model_type, {})
        model_class = self.MODEL_REGISTRY.get(self.model_type, XGBoostForecaster)
        model = model_class(**model_params)

        evaluator = ModelEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator, config=self.model_config)
        train_result = trainer.train(X_train, y_train, X_val, y_val)

        # Step 6: Evaluate on test set
        logger.info("Step 6: Evaluating on test set")
        test_predictions = model.predict(X_test)
        test_metrics = MetricsCalculator.compute_all(y_test, test_predictions)
        logger.info(f"Test metrics: {test_metrics}")

        return PipelineResult(
            model=model,
            metrics=test_metrics,
            val_metrics=train_result.metrics,
            run_id=train_result.run_id,
            model_version=None,
            data_shape=df.shape,
            feature_names=feature_names,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = TrainingPipeline(model_type="xgboost")
    result = pipeline.run()
    print(f"\nPipeline complete!")
    print(f"Test RMSE: {result.metrics.get('rmse', 'N/A'):.4f}")
    print(f"Test MAE: {result.metrics.get('mae', 'N/A'):.4f}")
    print(f"Test R2: {result.metrics.get('r2', 'N/A'):.4f}")
