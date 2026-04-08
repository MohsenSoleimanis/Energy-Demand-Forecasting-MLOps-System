"""Integration tests for monitoring (drift detection and performance tracking)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from energy_forecast.monitoring.drift import DriftDetector
from energy_forecast.monitoring.performance import PerformanceMonitor

pytestmark = pytest.mark.integration


class TestDriftDetection:
    """Tests for the DriftDetector class."""

    def test_drift_detection_no_drift(self):
        rng = np.random.default_rng(42)
        n = 500
        reference = pd.DataFrame({
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(5, 2, n),
        })
        # Current data drawn from the same distribution
        current = pd.DataFrame({
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(5, 2, n),
        })

        detector = DriftDetector(reference)
        report = detector.detect_drift(current)
        # With same distribution, we expect no dataset-level drift
        assert report.is_drifted is False

    def test_drift_detection_with_drift(self):
        rng = np.random.default_rng(42)
        n = 500
        reference = pd.DataFrame({
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(5, 2, n),
        })
        # Current data with a significant shift
        current = pd.DataFrame({
            "feature_a": rng.normal(10, 1, n),  # Large mean shift
            "feature_b": rng.normal(20, 2, n),  # Large mean shift
        })

        detector = DriftDetector(reference)
        report = detector.detect_drift(current)
        assert report.is_drifted is True
        assert report.dataset_drift_score > 0

        # Individual features should show drift
        for feat_name, feat_result in report.feature_drift.items():
            assert feat_result.is_drifted == True  # noqa: E712


class TestPerformanceMonitor:
    """Tests for the PerformanceMonitor class."""

    def test_performance_monitor_tracking(self):
        monitor = PerformanceMonitor()
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0, 51.0])

        metrics = monitor.track_prediction(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

        history = monitor.get_performance_history()
        assert len(history) == 1

    def test_degradation_detection(self):
        monitor = PerformanceMonitor(config={"degradation_threshold": 0.1})

        # Track several predictions with increasing error
        rng = np.random.default_rng(42)
        y_true = rng.uniform(10, 100, 50)

        # Good predictions (baseline)
        y_pred_good = y_true + rng.normal(0, 1, 50)
        baseline_metrics = monitor.track_prediction(y_true, y_pred_good)

        # Bad predictions (degraded)
        y_pred_bad = y_true + rng.normal(0, 20, 50)
        monitor.track_prediction(y_true, y_pred_bad)

        current_metrics = monitor.compute_metrics()
        result = monitor.detect_degradation(baseline_metrics, current_metrics)

        # With much worse predictions, degradation should be detected
        assert result.is_degraded is True
        assert len(result.metric_deltas) > 0
