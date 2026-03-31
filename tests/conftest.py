"""Shared test fixtures for the Belgian Energy Demand Forecasting project."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_feature_base_df():
    """Create a realistic synthetic DataFrame matching the feature_base schema."""
    n_hours = 200
    base_time = datetime(2024, 1, 1)
    np.random.seed(42)

    timestamps = [base_time + timedelta(hours=i) for i in range(n_hours)]
    hours = np.array([t.hour for t in timestamps])

    # Realistic diurnal load pattern (MW)
    load = (
        9000
        + 2000 * np.sin(2 * np.pi * (hours - 6) / 24)
        + np.random.normal(0, 200, n_hours)
    )

    temperature = 15 + 10 * np.sin(2 * np.pi * (hours - 14) / 24) + np.random.normal(0, 1, n_hours)

    df = pd.DataFrame({
        "timestamp_brussels": timestamps,
        "load_mw": load,
        "price_eur_mwh": np.random.uniform(20, 200, n_hours),
        "temperature_2m": temperature,
        "feels_like_temp": temperature - 1,
        "wind_speed_10m": np.abs(np.random.normal(8, 4, n_hours)),
        "wind_direction_10m": np.random.uniform(0, 360, n_hours),
        "shortwave_radiation": np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 24)),
        "precipitation": np.random.exponential(1, n_hours),
        "cloud_cover": np.random.uniform(0, 100, n_hours),
        "pressure_msl": np.random.uniform(1000, 1025, n_hours),
        "renewable_share_pct": np.random.uniform(10, 60, n_hours),
        "nuclear_mw": np.random.uniform(3000, 5000, n_hours),
        "gas_mw": np.random.uniform(1000, 3000, n_hours),
        "is_belgian_holiday": np.random.choice([True, False], n_hours, p=[0.05, 0.95]),
        "is_weekend": [t.weekday() >= 5 for t in timestamps],
        "is_school_vacation": np.random.choice([True, False], n_hours, p=[0.2, 0.8]),
        "day_of_week": [t.weekday() for t in timestamps],
        "month": [t.month for t in timestamps],
        "hour_of_day": [t.hour for t in timestamps],
        "load_is_valid": [True] * n_hours,
        "weather_is_anomalous": [False] * n_hours,
    })

    return df


@pytest.fixture
def sample_training_df(sample_feature_base_df):
    """Create a sample training DataFrame with features and target."""
    from src.ml.features.engineering import prepare_features

    df = prepare_features(sample_feature_base_df.copy(), mode="training")
    df["target_load_24h"] = df["load_mw"].shift(-24)
    df = df.dropna(subset=["target_load_24h"]).reset_index(drop=True)
    return df


@pytest.fixture
def sample_prediction_request():
    """Sample valid prediction request data."""
    return {
        "timestamp_brussels": "2024-06-15T14:00:00",
        "temperature_2m": 22.5,
        "relative_humidity_2m": 65.0,
        "wind_speed_10m": 12.0,
        "wind_direction_10m": 180.0,
        "shortwave_radiation": 450.0,
        "precipitation": 0.0,
        "cloud_cover": 30.0,
        "pressure_msl": 1013.0,
    }
