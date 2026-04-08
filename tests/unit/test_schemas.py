"""Tests for Pydantic data schemas."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from energy_forecast.data.schemas import EnergyReading, PredictionResult

pytestmark = pytest.mark.unit


class TestEnergyReading:
    """Tests for the EnergyReading schema."""

    def test_energy_reading_valid(self):
        reading = EnergyReading(
            timestamp="2023-06-15T12:00:00",
            building_id="building_001",
            energy_demand_kwh=50.0,
            temperature=25.0,
            humidity=60.0,
            is_holiday=False,
            day_of_week=3,
        )
        assert reading.energy_demand_kwh == 50.0
        assert isinstance(reading.timestamp, datetime)
        assert reading.building_id == "building_001"

    def test_energy_reading_invalid(self):
        # energy_demand_kwh must be > 0
        with pytest.raises(ValidationError):
            EnergyReading(
                timestamp="2023-06-15T12:00:00",
                building_id="building_001",
                energy_demand_kwh=-10.0,
                temperature=25.0,
                humidity=60.0,
                is_holiday=False,
                day_of_week=3,
            )


class TestPredictionResult:
    """Tests for the PredictionResult schema."""

    def test_prediction_result_serialization(self):
        result = PredictionResult(
            timestamp="2023-06-15T12:00:00",
            building_id="building_001",
            predicted_kwh=50.0,
            confidence_lower=45.0,
            confidence_upper=55.0,
        )
        data = result.model_dump()
        assert "predicted_kwh" in data
        assert data["predicted_kwh"] == 50.0
        assert data["confidence_lower"] == 45.0
        assert data["confidence_upper"] == 55.0
        assert isinstance(data["timestamp"], datetime)
