"""Integration tests for FastAPI serving endpoints (SERVE-006)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow", reason="mlflow required for API integration tests")


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([9500.0])
    return model


@pytest.fixture
def mock_model_version():
    version = MagicMock()
    version.version = "1"
    version.name = "energy-demand-forecast"
    version.run_id = "test-run-id"
    return version


@pytest.fixture
def test_client(mock_model, mock_model_version):
    with patch("src.ml.serving.app.load_production_model") as mock_load:
        mock_load.return_value = (mock_model, mock_model_version)

        import src.ml.serving.app as app_module

        app_module._model = mock_model
        app_module._model_version = mock_model_version
        app_module._model_name = "energy-demand-forecast"

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app)
        yield client

        app_module._model = None
        app_module._model_version = None


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_valid_input(self, test_client, sample_prediction_request):
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_load_mw" in data
        assert "prediction_id" in data

    def test_predict_invalid_temperature(self, test_client, sample_prediction_request):
        sample_prediction_request["temperature_2m"] = 100.0
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, test_client):
        response = test_client.get("/metrics")
        assert response.status_code == 200
