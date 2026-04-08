"""Integration tests for the training pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from energy_forecast.data.processor import DataProcessor
from energy_forecast.data.synthetic import SyntheticDataGenerator
from energy_forecast.evaluation.evaluator import ModelEvaluator
from energy_forecast.models.linear import LinearForecaster
from energy_forecast.training.trainer import Trainer, TrainResult

pytestmark = pytest.mark.integration


def _setup_mlflow_mock(mock_mlflow, run_id="integration-test-run"):
    """Configure mock_mlflow with a proper context manager."""
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)


class TestTrainingPipeline:
    """Integration tests that run the full pipeline end to end."""

    @patch("energy_forecast.training.trainer.mlflow")
    def test_full_pipeline_runs(self, mock_mlflow):
        _setup_mlflow_mock(mock_mlflow, "integration-test-run")

        # Step 1: Generate synthetic data
        gen = SyntheticDataGenerator(
            num_buildings=1,
            start_date="2023-01-01",
            end_date="2023-02-28",
            random_seed=42,
        )
        df = gen.generate()
        assert len(df) > 0

        # Step 2: Process features
        processor = DataProcessor(
            config={
                "lag_features": [1, 2, 3],
                "rolling_windows": [3],
                "rolling_stats": ["mean"],
                "drop_na": True,
            }
        )
        df_features = processor.create_features(df)
        assert len(df_features) > 0

        # Step 3: Split data
        split = processor.split_data(df_features)
        assert split.X_train.shape[0] > 0
        assert split.X_val.shape[0] > 0
        assert split.X_test.shape[0] > 0

        # Step 4: Train model
        model = LinearForecaster(model_type="ridge", alpha=1.0)
        evaluator = ModelEvaluator()
        with patch.object(evaluator, "log_to_mlflow"):
            trainer = Trainer(model=model, evaluator=evaluator)
            result = trainer.train(
                split.X_train, split.y_train,
                split.X_val, split.y_val,
            )

        assert isinstance(result, TrainResult)
        assert result.model.is_fitted is True

    @patch("energy_forecast.training.trainer.mlflow")
    def test_pipeline_returns_valid_metrics(self, mock_mlflow):
        _setup_mlflow_mock(mock_mlflow, "metrics-test-run")

        gen = SyntheticDataGenerator(
            num_buildings=1,
            start_date="2023-01-01",
            end_date="2023-02-28",
            random_seed=42,
        )
        df = gen.generate()
        processor = DataProcessor(
            config={
                "lag_features": [1, 2, 3],
                "rolling_windows": [3],
                "rolling_stats": ["mean"],
                "drop_na": True,
            }
        )
        df_features = processor.create_features(df)
        split = processor.split_data(df_features)

        model = LinearForecaster(model_type="ridge", alpha=1.0)
        evaluator = ModelEvaluator()
        with patch.object(evaluator, "log_to_mlflow"):
            trainer = Trainer(model=model, evaluator=evaluator)
            result = trainer.train(
                split.X_train, split.y_train,
                split.X_val, split.y_val,
            )

        # Verify metrics are valid numeric values
        for key, value in result.metrics.items():
            assert isinstance(value, (int, float))
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"

        assert "mae" in result.metrics
        assert "rmse" in result.metrics
        assert result.metrics["mae"] >= 0
        assert result.metrics["rmse"] >= 0
