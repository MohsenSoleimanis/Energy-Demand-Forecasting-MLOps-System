"""Shared pytest fixtures for the Energy Demand Forecasting test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecast.data.synthetic import SyntheticDataGenerator
from energy_forecast.features.engineering import FeatureEngineer


@pytest.fixture(scope="session")
def sample_energy_df() -> pd.DataFrame:
    """Small DataFrame of synthetic energy data (2 buildings, 1 month)."""
    gen = SyntheticDataGenerator(
        num_buildings=2,
        start_date="2023-01-01",
        end_date="2023-01-31",
        frequency="h",
        random_seed=42,
    )
    return gen.generate()


@pytest.fixture(scope="session")
def sample_features_df(sample_energy_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with engineered features derived from sample_energy_df."""
    fe = FeatureEngineer()
    df = fe.create_time_features(sample_energy_df.copy())
    df = fe.create_weather_features(df)
    df = fe.create_lag_features(df, target_col="energy_demand_kwh", lags=[1, 2, 3])
    df = fe.create_rolling_features(df, target_col="energy_demand_kwh", windows=[3])
    df = df.dropna().reset_index(drop=True)
    return df


@pytest.fixture()
def sample_train_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tuple of (X_train, y_train, X_val, y_val) numpy arrays."""
    rng = np.random.default_rng(42)
    n_train, n_val, n_features = 200, 50, 10
    X_train = rng.standard_normal((n_train, n_features))
    y_train = X_train[:, 0] * 3.0 + X_train[:, 1] * 1.5 + rng.normal(0, 0.1, n_train)
    X_val = rng.standard_normal((n_val, n_features))
    y_val = X_val[:, 0] * 3.0 + X_val[:, 1] * 1.5 + rng.normal(0, 0.1, n_val)
    return X_train, y_train, X_val, y_val


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory fixture (alias for tmp_path)."""
    return tmp_path


@pytest.fixture()
def model_config() -> dict:
    """Sample model configuration dict."""
    return {
        "models": {
            "linear": {"type": "ridge", "alpha": 1.0},
            "xgboost": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.1,
                "early_stopping_rounds": 10,
            },
        },
        "training": {
            "cross_validation": {
                "n_splits": 3,
                "strategy": "expanding_window",
                "gap": 0,
            }
        },
    }


@pytest.fixture()
def data_config() -> dict:
    """Sample data configuration dict."""
    return {
        "synthetic": {
            "num_buildings": 2,
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "frequency": "h",
            "random_seed": 42,
            "noise_level": 0.05,
        },
        "processing": {
            "lag_features": [1, 2, 3, 24],
            "rolling_windows": [3, 24],
            "target_column": "energy_demand_kwh",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "normalization": "standard",
        },
    }
