"""Tests for MetricsCalculator."""

from __future__ import annotations

import numpy as np
import pytest

from energy_forecast.evaluation.metrics import MetricsCalculator

pytestmark = pytest.mark.unit


class TestMetricsCalculator:
    """Tests for the MetricsCalculator class."""

    def test_mae_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        result = MetricsCalculator.mae(y_true, y_pred)
        assert abs(result - 0.5) < 1e-10

    def test_rmse_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = MetricsCalculator.rmse(y_true, y_pred)
        assert result == 0.0

        y_pred2 = np.array([2.0, 3.0, 4.0])
        result2 = MetricsCalculator.rmse(y_true, y_pred2)
        assert abs(result2 - 1.0) < 1e-10

    def test_mape_known_values(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 330.0])
        result = MetricsCalculator.mape(y_true, y_pred)
        # |10/100| + |10/200| + |30/300| = 0.1 + 0.05 + 0.1 = 0.25 -> 25/3 %
        expected = (10 / 100 + 10 / 200 + 30 / 300) / 3 * 100
        assert abs(result - expected) < 1e-10

    def test_r2_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        result = MetricsCalculator.r2(y_true, y_pred)
        assert abs(result - 1.0) < 1e-10

    def test_compute_all_returns_all_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 18.0, 33.0, 37.0, 52.0])
        result = MetricsCalculator.compute_all(y_true, y_pred)
        expected_keys = {"mae", "rmse", "mape", "r2", "peak_hour_error"}
        assert set(result.keys()) == expected_keys
        for key, value in result.items():
            assert isinstance(value, float)
