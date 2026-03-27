"""Unit tests for Pydantic schemas (SERVE-005)."""

import pytest
from datetime import datetime

from src.ml.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse,
)


class TestPredictionRequest:
    def test_valid_request(self, sample_prediction_request):
        req = PredictionRequest(**sample_prediction_request)
        assert req.temperature_2m == 22.5
        assert req.wind_speed_10m == 12.0

    def test_temperature_out_of_range(self, sample_prediction_request):
        sample_prediction_request["temperature_2m"] = 60.0
        with pytest.raises(Exception):
            PredictionRequest(**sample_prediction_request)

    def test_negative_wind_speed(self, sample_prediction_request):
        sample_prediction_request["wind_speed_10m"] = -5.0
        with pytest.raises(Exception):
            PredictionRequest(**sample_prediction_request)

    def test_humidity_over_100(self, sample_prediction_request):
        sample_prediction_request["relative_humidity_2m"] = 101.0
        with pytest.raises(Exception):
            PredictionRequest(**sample_prediction_request)

    def test_optional_fields(self, sample_prediction_request):
        req = PredictionRequest(**sample_prediction_request)
        assert req.price_eur_mwh is None
        assert req.load_lag_1h is None

    def test_optional_fields_with_values(self, sample_prediction_request):
        sample_prediction_request["price_eur_mwh"] = 55.0
        sample_prediction_request["load_lag_24h"] = 9500.0
        req = PredictionRequest(**sample_prediction_request)
        assert req.price_eur_mwh == 55.0
        assert req.load_lag_24h == 9500.0

    def test_pressure_out_of_range(self, sample_prediction_request):
        sample_prediction_request["pressure_msl"] = 800.0
        with pytest.raises(Exception):
            PredictionRequest(**sample_prediction_request)


class TestPredictionResponse:
    def test_response_creation(self):
        resp = PredictionResponse(
            timestamp_brussels=datetime(2024, 6, 15, 14, 0),
            predicted_load_mw=9500.0,
            model_version="1",
        )
        assert resp.predicted_load_mw == 9500.0
        assert resp.prediction_id is not None
        assert resp.predicted_at is not None


class TestBatchPredictionRequest:
    def test_batch_within_limit(self, sample_prediction_request):
        batch = BatchPredictionRequest(
            predictions=[PredictionRequest(**sample_prediction_request)] * 10
        )
        assert len(batch.predictions) == 10

    def test_batch_exceeds_limit(self, sample_prediction_request):
        with pytest.raises(Exception):
            BatchPredictionRequest(
                predictions=[PredictionRequest(**sample_prediction_request)] * 1001
            )


class TestHealthResponse:
    def test_health_response(self):
        resp = HealthResponse(status="healthy", model_version="3", model_alias="production")
        assert resp.status == "healthy"
