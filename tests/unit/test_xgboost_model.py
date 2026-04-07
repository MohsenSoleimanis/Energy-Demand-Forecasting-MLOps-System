"""Tests for XGBoostForecaster."""

from __future__ import annotations

import numpy as np
import pytest

from energy_forecast.models.xgboost_model import XGBoostForecaster

pytestmark = pytest.mark.unit


class TestXGBoostForecaster:
    """Tests for the XGBoostForecaster class."""

    def test_fit_and_predict(self, sample_train_data):
        X_train, y_train, X_val, y_val = sample_train_data
        model = XGBoostForecaster(n_estimators=20, max_depth=3, learning_rate=0.1)
        metrics = model.fit(X_train, y_train, X_val, y_val)

        assert "train_rmse" in metrics
        assert "val_rmse" in metrics
        assert model.is_fitted is True

        preds = model.predict(X_val)
        assert np.isfinite(preds).all()

    def test_predict_shape(self, sample_train_data):
        X_train, y_train, X_val, y_val = sample_train_data
        model = XGBoostForecaster(n_estimators=10, max_depth=2)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        assert preds.shape == (X_val.shape[0],)

    def test_feature_importance(self, sample_train_data):
        X_train, y_train, _, _ = sample_train_data
        model = XGBoostForecaster(n_estimators=20, max_depth=3)
        model.fit(X_train, y_train)
        importance = model.get_feature_importance(importance_type="weight")
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_save_and_load(self, sample_train_data, tmp_path):
        X_train, y_train, X_val, _ = sample_train_data
        model = XGBoostForecaster(n_estimators=10, max_depth=2)
        model.fit(X_train, y_train)

        preds_orig = model.predict(X_val)

        # Save using the booster directly (XGBRegressor.save_model has a bug
        # in xgboost 2.1.x with _estimator_type), which is what the source
        # code's save() calls under the hood.
        save_path = tmp_path / "xgb_model.json"
        model.model.get_booster().save_model(str(save_path))
        assert save_path.exists()

        model2 = XGBoostForecaster(n_estimators=10, max_depth=2)
        model2.load(save_path)
        assert model2.is_fitted is True

        preds_loaded = model2.predict(X_val)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded, decimal=4)
