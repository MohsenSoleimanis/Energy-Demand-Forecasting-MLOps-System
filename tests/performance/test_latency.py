"""Performance benchmark tests for model prediction latency."""

from __future__ import annotations

import time

import numpy as np
import pytest

from energy_forecast.models.linear import LinearForecaster
from energy_forecast.models.xgboost_model import XGBoostForecaster

pytestmark = pytest.mark.performance


class TestPredictionLatency:
    """Performance benchmarks for prediction latency."""

    def test_prediction_latency(self, sample_train_data):
        """Model.predict should complete in under 100ms for a single sample."""
        X_train, y_train, X_val, _ = sample_train_data
        model = LinearForecaster(model_type="ridge", alpha=1.0)
        model.fit(X_train, y_train)

        single_sample = X_val[:1]

        # Warm up
        model.predict(single_sample)

        start = time.perf_counter()
        for _ in range(100):
            model.predict(single_sample)
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.1, f"Single prediction took {elapsed:.4f}s (limit: 0.1s)"

    def test_batch_throughput(self, sample_train_data):
        """XGBoost batch prediction throughput test."""
        X_train, y_train, _, _ = sample_train_data
        model = XGBoostForecaster(n_estimators=50, max_depth=3)
        model.fit(X_train, y_train)

        # Create a larger batch
        rng = np.random.default_rng(42)
        X_batch = rng.standard_normal((1000, X_train.shape[1]))

        # Warm up
        model.predict(X_batch)

        start = time.perf_counter()
        preds = model.predict(X_batch)
        elapsed = time.perf_counter() - start

        assert preds.shape == (1000,)
        # Batch of 1000 should complete well under 1 second
        assert elapsed < 1.0, f"Batch prediction took {elapsed:.4f}s (limit: 1.0s)"
        throughput = 1000 / elapsed
        # Should process at least 1000 samples per second
        assert throughput > 1000, f"Throughput: {throughput:.0f} samples/s (min: 1000)"
