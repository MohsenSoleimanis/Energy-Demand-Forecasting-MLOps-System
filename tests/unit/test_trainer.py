"""Tests for Trainer (with mocked MLflow)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from energy_forecast.evaluation.evaluator import ModelEvaluator
from energy_forecast.models.linear import LinearForecaster
from energy_forecast.training.trainer import Trainer, TrainResult

pytestmark = pytest.mark.unit


def _make_mlflow_mock():
    """Create a properly configured MLflow mock."""
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run
    return mock_mlflow


class TestTrainer:
    """Tests for the Trainer class."""

    @patch("energy_forecast.training.trainer.mlflow")
    def test_train_returns_result(self, mock_mlflow, sample_train_data):
        X_train, y_train, X_val, y_val = sample_train_data

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = LinearForecaster(model_type="ridge", alpha=1.0)
        evaluator = ModelEvaluator()

        # Patch the evaluator's log_to_mlflow to avoid real MLflow interaction
        with patch.object(evaluator, "log_to_mlflow"):
            trainer = Trainer(model=model, evaluator=evaluator)
            result = trainer.train(X_train, y_train, X_val, y_val)

        assert isinstance(result, TrainResult)
        assert result.run_id == "test-run-123"
        assert isinstance(result.metrics, dict)
        assert "mae" in result.metrics
        assert "rmse" in result.metrics
        assert result.model.is_fitted is True

    @patch("energy_forecast.training.trainer.mlflow")
    def test_train_logs_metrics(self, mock_mlflow, sample_train_data):
        X_train, y_train, X_val, y_val = sample_train_data

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-456"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = LinearForecaster(model_type="ridge", alpha=1.0)
        evaluator = ModelEvaluator()

        with patch.object(evaluator, "log_to_mlflow"):
            trainer = Trainer(model=model, evaluator=evaluator)
            trainer.train(X_train, y_train, X_val, y_val)

        # Verify MLflow logging was called
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metric.assert_called()
