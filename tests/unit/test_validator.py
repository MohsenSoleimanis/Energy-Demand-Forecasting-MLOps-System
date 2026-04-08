"""Tests for DataValidator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.data.validator import DataValidator

pytestmark = pytest.mark.unit


class TestDataValidator:
    """Tests for the DataValidator class."""

    def _make_valid_df(self) -> pd.DataFrame:
        """Create a small valid DataFrame for testing."""
        n = 24
        return pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
            "building_id": "building_001",
            "energy_demand_kwh": np.random.default_rng(42).uniform(10, 100, n),
            "temperature": np.random.default_rng(42).uniform(0, 30, n),
            "humidity": np.random.default_rng(42).uniform(30, 80, n),
            "is_holiday": False,
            "day_of_week": pd.date_range("2023-01-01", periods=n, freq="h").dayofweek,
        })

    def test_valid_data_passes(self):
        validator = DataValidator()
        df = self._make_valid_df()
        result = validator.validate_raw_data(df)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_detects_missing_columns(self):
        validator = DataValidator()
        df = self._make_valid_df().drop(columns=["temperature"])
        result = validator.validate_raw_data(df)
        assert result.is_valid is False
        assert any("Missing required columns" in e for e in result.errors)

    def test_detects_negative_energy(self):
        validator = DataValidator()
        df = self._make_valid_df()
        df.loc[0, "energy_demand_kwh"] = -5.0
        result = validator.validate_raw_data(df)
        assert result.is_valid is False
        assert any("non-positive" in e for e in result.errors)

    def test_detects_null_values(self):
        validator = DataValidator()
        df = self._make_valid_df()
        df.loc[0, "energy_demand_kwh"] = np.nan
        result = validator.validate_raw_data(df)
        assert result.is_valid is False
        assert any("null" in e.lower() for e in result.errors)
