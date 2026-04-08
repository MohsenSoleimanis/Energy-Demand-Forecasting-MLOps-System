"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from energy_forecast.serving.routes import router
from energy_forecast.serving.dependencies import ModelManager

pytestmark = pytest.mark.integration


def _create_test_app() -> FastAPI:
    """Create a minimal test FastAPI app with the prediction router."""
    app = FastAPI()
    app.include_router(router)
    return app


class _MockModel:
    """Mock model that returns deterministic predictions."""

    def predict(self, X):
        return np.full(len(X), 42.0)


@pytest.fixture()
def test_client():
    """Create a test client with a mocked model manager."""
    app = _create_test_app()

    mock_manager = MagicMock(spec=ModelManager)
    mock_manager.is_loaded = True
    mock_manager.get_model.return_value = _MockModel()
    mock_manager.get_model_info.return_value = {
        "model_name": "test-model",
        "model_version": "1.0.0",
        "model_stage": "Production",
        "metrics": {"rmse": 5.0, "mae": 3.0},
        "features_used": ["temperature", "humidity", "hour_sin", "hour_cos"],
    }

    from energy_forecast.serving import dependencies as deps
    app.dependency_overrides[deps.get_model_manager] = lambda: mock_manager

    return TestClient(app)


class TestAPIEndpoints:
    """Tests for the FastAPI API endpoints."""

    def test_health_endpoint(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    @patch("energy_forecast.serving.routes.track_prediction_metrics")
    def test_predict_endpoint(self, mock_track, test_client):
        payload = {
            "timestamp": "2023-06-15T12:00:00",
            "building_id": "building_001",
            "temperature": 25.0,
            "humidity": 60.0,
        }
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_kwh" in data
        assert "confidence_lower" in data
        assert "confidence_upper" in data
        assert data["building_id"] == "building_001"

    def test_model_info_endpoint(self, test_client):
        response = test_client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert data["model_version"] == "1.0.0"
        assert "features_used" in data
        assert isinstance(data["features_used"], list)
