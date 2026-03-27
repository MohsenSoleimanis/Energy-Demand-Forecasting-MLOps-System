"""
Feature engineering module for Belgian Energy Demand Forecasting.

Pure functional module: all functions take a DataFrame and return a DataFrame.
No database or API calls. Imported by BOTH training AND serving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features that use only past information."""
    if df.empty:
        return df
    df = df.copy()
    if "load_mw" in df.columns:
        df["load_lag_1h"] = df["load_mw"].shift(1)
        df["load_lag_24h"] = df["load_mw"].shift(24)
        df["load_lag_168h"] = df["load_mw"].shift(168)
    if "price_eur_mwh" in df.columns:
        df["price_lag_24h"] = df["price_eur_mwh"].shift(24)
    if "temperature_2m" in df.columns:
        df["temp_lag_24h"] = df["temperature_2m"].shift(24)
    return df


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window statistics. Window looks backward only."""
    if df.empty:
        return df
    df = df.copy()
    if "load_mw" in df.columns:
        df["load_rolling_mean_24h"] = (
            df["load_mw"].rolling(window=24, min_periods=1).mean()
        )
        df["load_rolling_std_24h"] = (
            df["load_mw"].rolling(window=24, min_periods=1).std()
        )
        df["load_rolling_mean_168h"] = (
            df["load_mw"].rolling(window=168, min_periods=1).mean()
        )
    if "temperature_2m" in df.columns:
        df["temp_rolling_mean_24h"] = (
            df["temperature_2m"].rolling(window=24, min_periods=1).mean()
        )
    return df


# ---------------------------------------------------------------------------
# Cyclical features
# ---------------------------------------------------------------------------

def compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour_of_day and month as sin/cos pairs."""
    if df.empty:
        return df
    df = df.copy()

    # Determine hour
    if "hour_of_day" in df.columns:
        hour = df["hour_of_day"].astype(float)
    elif "timestamp_brussels" in df.columns:
        ts = pd.to_datetime(df["timestamp_brussels"])
        hour = ts.dt.hour + ts.dt.minute / 60.0
    else:
        return df

    # Determine month
    if "month" in df.columns:
        month = df["month"].astype(float)
    elif "timestamp_brussels" in df.columns:
        ts = pd.to_datetime(df["timestamp_brussels"])
        month = ts.dt.month.astype(float)
    else:
        month = pd.Series([1.0] * len(df))

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute temp_x_hour, wind_x_radiation."""
    if df.empty:
        return df
    df = df.copy()

    if "temperature_2m" in df.columns and "hour_of_day" in df.columns:
        df["temp_x_hour"] = df["temperature_2m"] * df["hour_of_day"]
    elif "temperature_2m" in df.columns and "timestamp_brussels" in df.columns:
        ts = pd.to_datetime(df["timestamp_brussels"])
        df["temp_x_hour"] = df["temperature_2m"] * ts.dt.hour

    if "wind_speed_10m" in df.columns and "shortwave_radiation" in df.columns:
        df["wind_x_radiation"] = df["wind_speed_10m"] * df["shortwave_radiation"]

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, mode: str = "training") -> pd.DataFrame:
    """Main feature-engineering entry point.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns matching gold.feature_base schema.
    mode : str
        "training" uses weather actuals, "serving" expects weather forecast
        columns (same names). Both produce identical output columns.
    """
    if df.empty:
        return df
    df = compute_lag_features(df)
    df = compute_rolling_features(df)
    df = compute_cyclical_features(df)
    df = compute_interaction_features(df)
    return df


# ---------------------------------------------------------------------------
# Feature column list
# ---------------------------------------------------------------------------

_FEATURE_COLUMNS: list[str] = [
    # --- raw / base ---
    "load_mw",
    "price_eur_mwh",
    "temperature_2m",
    "feels_like_temp",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "renewable_share_pct",
    "nuclear_mw",
    "gas_mw",
    "is_belgian_holiday",
    "is_weekend",
    "is_school_vacation",
    "day_of_week",
    "month",
    "hour_of_day",
    # --- lag ---
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "price_lag_24h",
    "temp_lag_24h",
    # --- rolling ---
    "load_rolling_mean_24h",
    "load_rolling_std_24h",
    "load_rolling_mean_168h",
    "temp_rolling_mean_24h",
    # --- cyclical ---
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # --- interaction ---
    "temp_x_hour",
    "wind_x_radiation",
]


def get_feature_columns() -> list[str]:
    """Return the canonical ordered list of feature column names."""
    return list(_FEATURE_COLUMNS)
