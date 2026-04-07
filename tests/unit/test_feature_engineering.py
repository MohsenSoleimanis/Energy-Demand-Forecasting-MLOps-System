"""Tests for FeatureEngineer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.features.engineering import FeatureEngineer

pytestmark = pytest.mark.unit


def _make_simple_df(n: int = 48) -> pd.DataFrame:
    """Create a small DataFrame with a timestamp column and numeric target."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-06-01", periods=n, freq="h"),
        "energy_demand": np.arange(1.0, n + 1.0),
        "temperature": np.random.default_rng(42).uniform(10, 35, n),
        "humidity": np.random.default_rng(42).uniform(30, 80, n),
    })


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_create_time_features_adds_columns(self):
        fe = FeatureEngineer()
        df = _make_simple_df()
        result = fe.create_time_features(df)
        expected_cols = {
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
            "day_of_year_sin", "day_of_year_cos",
        }
        assert expected_cols.issubset(set(result.columns))
        # Values should be in [-1, 1]
        for col in expected_cols:
            assert result[col].min() >= -1.0 - 1e-10
            assert result[col].max() <= 1.0 + 1e-10

    def test_lag_features_correct_values(self):
        fe = FeatureEngineer()
        df = _make_simple_df(n=10)
        result = fe.create_lag_features(df, target_col="energy_demand", lags=[1, 2])
        # lag_1 of row 1 should equal row 0's value
        assert result["energy_demand_lag_1"].iloc[1] == df["energy_demand"].iloc[0]
        # lag_2 of row 2 should equal row 0's value
        assert result["energy_demand_lag_2"].iloc[2] == df["energy_demand"].iloc[0]
        # lag_1 of row 0 should be NaN
        assert pd.isna(result["energy_demand_lag_1"].iloc[0])

    def test_rolling_features_correct_values(self):
        fe = FeatureEngineer()
        df = _make_simple_df(n=10)
        result = fe.create_rolling_features(
            df, target_col="energy_demand", windows=[3]
        )
        # Rolling mean of window 3 at index 2 should be mean(1,2,3) = 2.0
        assert "energy_demand_rolling_mean_3" in result.columns
        val = result["energy_demand_rolling_mean_3"].iloc[2]
        expected = np.mean([1.0, 2.0, 3.0])
        assert abs(val - expected) < 1e-10

        # Rolling max at index 4 should be 5.0 (values 3,4,5)
        assert result["energy_demand_rolling_max_3"].iloc[4] == 5.0

    def test_weather_features(self):
        fe = FeatureEngineer()
        df = _make_simple_df()
        result = fe.create_weather_features(df)
        assert "heating_degree_days" in result.columns
        assert "cooling_degree_days" in result.columns
        assert "temp_humidity_interaction" in result.columns
        # Heating degree days should be non-negative
        assert (result["heating_degree_days"] >= 0).all()
        assert (result["cooling_degree_days"] >= 0).all()

    def test_calendar_features(self):
        fe = FeatureEngineer()
        df = _make_simple_df(n=168)  # 1 week
        result = fe.create_calendar_features(df)
        assert "is_weekend" in result.columns
        assert "is_business_hour" in result.columns
        assert "season" in result.columns
        assert "is_holiday" in result.columns
        # is_weekend should be 0 or 1
        assert set(result["is_weekend"].unique()).issubset({0, 1})
