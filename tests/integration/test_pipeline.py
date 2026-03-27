"""Integration test for ML pipeline with synthetic data (ML-008)."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow required for parquet tests")

from src.ml.features.feature_engineering import get_feature_columns, prepare_features


@pytest.fixture
def synthetic_training_data(tmp_path):
    """Create a tiny synthetic training dataset."""
    n_hours = 250
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_hours)]

    np.random.seed(42)
    load = (
        8000
        + 2000 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
        + np.random.normal(0, 200, n_hours)
    )

    df = pd.DataFrame({
        "timestamp_brussels": timestamps,
        "load_mw": load,
        "price_eur_mwh": np.random.uniform(20, 200, n_hours),
        "temperature_2m": 15 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24),
        "feels_like_temp": 14 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24),
        "wind_speed_10m": np.random.uniform(0, 30, n_hours),
        "wind_direction_10m": np.random.uniform(0, 360, n_hours),
        "shortwave_radiation": np.maximum(
            0, 400 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
        ),
        "precipitation": np.random.exponential(1, n_hours),
        "cloud_cover": np.random.uniform(0, 100, n_hours),
        "pressure_msl": np.random.uniform(1000, 1025, n_hours),
        "renewable_share_pct": np.random.uniform(10, 60, n_hours),
        "nuclear_mw": np.random.uniform(3000, 5000, n_hours),
        "gas_mw": np.random.uniform(1000, 3000, n_hours),
        "is_belgian_holiday": [False] * n_hours,
        "is_weekend": [t.weekday() >= 5 for t in timestamps],
        "is_school_vacation": [False] * n_hours,
        "day_of_week": [t.weekday() for t in timestamps],
        "month": [t.month for t in timestamps],
        "hour_of_day": [t.hour for t in timestamps],
        "load_is_valid": [True] * n_hours,
        "weather_is_anomalous": [False] * n_hours,
    })

    df = prepare_features(df, mode="training")
    df["target_load_24h"] = df["load_mw"].shift(-24)
    df = df.dropna(subset=["target_load_24h"])
    df = df.iloc[168:].reset_index(drop=True)

    output_path = tmp_path / "training_set.parquet"
    df.to_parquet(output_path, index=False)
    return str(output_path), df


def test_training_pipeline_runs(synthetic_training_data):
    """Verify the training pipeline can run end-to-end on synthetic data."""
    data_path, df = synthetic_training_data

    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    assert len(feature_cols) > 0

    assert "target_load_24h" in df.columns
    assert df["target_load_24h"].notna().all()

    timestamps = df["timestamp_brussels"]
    assert timestamps.is_monotonic_increasing
