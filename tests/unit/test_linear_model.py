"""Tests for LinearForecaster."""

from __future__ import annotations

import numpy as np
import pytest

from energy_forecast.models.linear import LinearForecaster

pytestmark = pytest.mark.unit


class TestLinearForecaster:
    """Tests for the LinearForecaster class."""

    def test_fit_and_predict(self, sample_train_data):
        X_train, y_train, X_val, y_val = sample_train_data
        model = LinearForecaster(model_type="ridge", alpha=1.0)
        metrics = model.fit(X_train, y_train, X_val, y_val)

        assert "train_rmse" in metrics
        assert "val_rmse" in metrics
        assert model.is_fitted is True

        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)
        assert np.isfinite(preds).all()

    def test_save_and_load(self, sample_train_data, tmp_path):
        X_train, y_train, X_val, y_val = sample_train_data
        model = LinearForecaster(model_type="ridge", alpha=1.0)
        model.fit(X_train, y_train)

        save_path = tmp_path / "model.joblib"
        model.save(save_path)
        assert save_path.exists()

        model2 = LinearForecaster(model_type="ridge", alpha=1.0)
        model2.load(save_path)
        assert model2.is_fitted is True

        preds_orig = model.predict(X_val)
        preds_loaded = model2.predict(X_val)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)

    def test_get_params(self):
        model = LinearForecaster(model_type="lasso", alpha=0.5)
        params = model.get_params()
        assert params["model_type"] == "lasso"
        assert params["alpha"] == 0.5

    def test_get_feature_importance(self, sample_train_data):
        X_train, y_train, _, _ = sample_train_data
        model = LinearForecaster(model_type="ridge", alpha=1.0)
        model.fit(X_train, y_train)
        importance = model.get_feature_importance()
        assert isinstance(importance, np.ndarray)
        assert len(importance) == X_train.shape[1]
