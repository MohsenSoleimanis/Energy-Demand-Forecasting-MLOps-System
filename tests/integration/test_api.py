"""Integration tests for FastAPI serving endpoints (SERVE-006)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow", reason="mlflow required for API integration tests")

TEST_API_KEY = "test-secret-key-12345"


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
    os.environ["API_KEYS"] = TEST_API_KEY
    with patch("src.ml.serving.app.load_production_model") as mock_load:
        mock_load.return_value = (mock_model, mock_model_version)

        import src.ml.serving.app as app_module

        app_module._model = mock_model
        app_module._model_version = mock_model_version
        app_module._model_name = "energy-demand-forecast"
        app_module._prediction_logger = None

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app)
        yield client

        app_module._model = None
        app_module._model_version = None
        app_module._prediction_logger = None

    os.environ.pop("API_KEYS", None)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_no_auth_required(self, test_client):
        """Health endpoint should work without an API key."""
        response = test_client.get("/health")
        assert response.status_code == 200


class TestPredictEndpoint:
    def test_predict_valid_input(self, test_client, sample_prediction_request, auth_headers):
        response = test_client.post("/predict", json=sample_prediction_request, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_load_mw" in data
        assert "prediction_id" in data

    def test_predict_invalid_temperature(self, test_client, sample_prediction_request, auth_headers):
        sample_prediction_request["temperature_2m"] = 100.0
        response = test_client.post("/predict", json=sample_prediction_request, headers=auth_headers)
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, test_client):
        response = test_client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_no_auth_required(self, test_client):
        """Metrics endpoint should work without an API key."""
        response = test_client.get("/metrics")
        assert response.status_code == 200


class TestAuthEndpoints:
    def test_predict_without_key_returns_401(self, test_client, sample_prediction_request):
        """Predict without API key should return 401."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 401

    def test_predict_wrong_key_returns_403(self, test_client, sample_prediction_request):
        """Predict with wrong API key should return 403."""
        response = test_client.post(
            "/predict",
            json=sample_prediction_request,
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403
