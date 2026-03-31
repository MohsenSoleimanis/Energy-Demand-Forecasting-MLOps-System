"""Feature engineering module for Belgian Energy Demand Forecasting.

Pure functional module: all functions take a DataFrame and return a DataFrame.
No database or API calls. Imported by BOTH training AND serving.

Window sizes and the canonical feature column list are loaded from
``configs/data/features.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.shared.config import load_config

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "features.yaml"


def _load_features_config() -> dict[str, Any]:
    """Load and return the feature engineering configuration.

    Raises:
        ConfigError: If the config file is missing or malformed.
    """
    return load_config(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features using only past information.

    Lag windows are read from ``configs/data/features.yaml``. Missing source
    columns trigger a logged warning (never a silent default).

    Args:
        df: Input DataFrame (must be sorted by time).

    Returns:
        DataFrame with lag columns appended.
    """
    if df.empty:
        return df
    df = df.copy()
    cfg = _load_features_config()
    lag_cfg = cfg["lag_features"]

    _apply_lags(df, "load_mw", "load_lag", lag_cfg.get("load_lags", []))
    _apply_lags(df, "price_eur_mwh", "price_lag", lag_cfg.get("price_lags", []))
    _apply_lags(df, "temperature_2m", "temp_lag", lag_cfg.get("temp_lags", []))
    return df


def _apply_lags(
    df: pd.DataFrame,
    source_col: str,
    prefix: str,
    lags: list[int],
) -> None:
    """Apply a list of lag shifts from *source_col*, mutating *df* in place.

    Args:
        df: DataFrame to mutate.
        source_col: Column to shift.
        prefix: Naming prefix (e.g. ``"load_lag"``).
        lags: List of lag periods in hours.
    """
    if source_col not in df.columns:
        if lags:
            logger.warning(
                "Column '%s' not found -- skipping lags %s", source_col, lags,
            )
        return
    for lag in lags:
        df[f"{prefix}_{lag}h"] = df[source_col].shift(lag)


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window statistics (backward-looking only).

    Window sizes and ``min_periods`` are read from
    ``configs/data/features.yaml``.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with rolling statistic columns appended.
    """
    if df.empty:
        return df
    df = df.copy()
    cfg = _load_features_config()
    roll_cfg = cfg["rolling_features"]
    min_periods: int = roll_cfg.get("min_periods", 1)

    _apply_rolling(df, "load_mw", "load_rolling", roll_cfg.get("load_windows", []), min_periods)
    _apply_rolling(df, "temperature_2m", "temp_rolling", roll_cfg.get("temp_windows", []), min_periods)
    return df


def _apply_rolling(
    df: pd.DataFrame,
    source_col: str,
    prefix: str,
    windows: list[int],
    min_periods: int,
) -> None:
    """Compute rolling mean and std from *source_col*, mutating *df* in place.

    Args:
        df: DataFrame to mutate.
        source_col: Column to compute rolling stats on.
        prefix: Naming prefix (e.g. ``"load_rolling"``).
        windows: List of window sizes in hours.
        min_periods: Minimum observations required for a valid result.
    """
    if source_col not in df.columns:
        if windows:
            logger.warning(
                "Column '%s' not found -- skipping rolling windows %s",
                source_col, windows,
            )
        return
    for w in windows:
        roller = df[source_col].rolling(window=w, min_periods=min_periods)
        df[f"{prefix}_mean_{w}h"] = roller.mean()
        df[f"{prefix}_std_{w}h"] = roller.std()


# ---------------------------------------------------------------------------
# Cyclical features
# ---------------------------------------------------------------------------

def compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ``hour_of_day`` and ``month`` as sin/cos pairs.

    Periods are read from ``configs/data/features.yaml``.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with ``hour_sin``, ``hour_cos``, ``month_sin``,
        ``month_cos`` columns appended.
    """
    if df.empty:
        return df
    df = df.copy()
    cfg = _load_features_config()
    cyc_cfg = cfg["cyclical_features"]
    hour_period: int = cyc_cfg["hour_period"]
    month_period: int = cyc_cfg["month_period"]

    hour = _resolve_hour(df)
    month = _resolve_month(df)

    df["hour_sin"] = np.sin(2 * np.pi * hour / hour_period)
    df["hour_cos"] = np.cos(2 * np.pi * hour / hour_period)
    df["month_sin"] = np.sin(2 * np.pi * month / month_period)
    df["month_cos"] = np.cos(2 * np.pi * month / month_period)
    return df


def _resolve_hour(df: pd.DataFrame) -> pd.Series:
    """Extract hour values from *df*, preferring explicit column over timestamp.

    Args:
        df: Source DataFrame.

    Returns:
        Series of float hour values.

    Raises:
        KeyError: If neither ``hour_of_day`` nor ``timestamp_brussels`` exist.
    """
    if "hour_of_day" in df.columns:
        return df["hour_of_day"].astype(float)
    if "timestamp_brussels" in df.columns:
        ts = pd.to_datetime(df["timestamp_brussels"])
        return ts.dt.hour + ts.dt.minute / 60.0
    raise KeyError(
        "Cannot compute cyclical hour features: "
        "neither 'hour_of_day' nor 'timestamp_brussels' found in DataFrame"
    )


def _resolve_month(df: pd.DataFrame) -> pd.Series:
    """Extract month values from *df*, preferring explicit column over timestamp.

    Args:
        df: Source DataFrame.

    Returns:
        Series of float month values.

    Raises:
        KeyError: If neither ``month`` nor ``timestamp_brussels`` exist.
    """
    if "month" in df.columns:
        return df["month"].astype(float)
    if "timestamp_brussels" in df.columns:
        ts = pd.to_datetime(df["timestamp_brussels"])
        return ts.dt.month.astype(float)
    raise KeyError(
        "Cannot compute cyclical month features: "
        "neither 'month' nor 'timestamp_brussels' found in DataFrame"
    )


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction terms defined in ``configs/data/features.yaml``.

    Each interaction is a pair of column names. The product is stored under a
    descriptive name. Missing columns produce a logged warning.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with interaction columns appended.
    """
    if df.empty:
        return df
    df = df.copy()
    cfg = _load_features_config()
    interactions: list[list[str]] = cfg.get("interaction_features", [])

    # Map column pairs to output names
    _NAMES = {
        ("temperature_2m", "hour_of_day"): "temp_x_hour",
        ("wind_speed_10m", "shortwave_radiation"): "wind_x_radiation",
    }

    for pair in interactions:
        col_a, col_b = pair[0], pair[1]
        out_name = _NAMES.get((col_a, col_b), f"{col_a}_x_{col_b}")

        # Allow hour_of_day fallback to timestamp_brussels
        a_vals = _get_column_or_derived(df, col_a)
        b_vals = _get_column_or_derived(df, col_b)

        if a_vals is None or b_vals is None:
            missing = [c for c, v in [(col_a, a_vals), (col_b, b_vals)] if v is None]
            logger.warning(
                "Missing columns %s -- skipping interaction '%s'", missing, out_name,
            )
            continue
        df[out_name] = a_vals * b_vals

    return df


def _get_column_or_derived(
    df: pd.DataFrame, col: str,
) -> pd.Series | None:
    """Return *col* from *df*, or derive it from timestamp if possible.

    Args:
        df: Source DataFrame.
        col: Column name to retrieve.

    Returns:
        Series if available, ``None`` otherwise.
    """
    if col in df.columns:
        return df[col]
    if col == "hour_of_day" and "timestamp_brussels" in df.columns:
        return pd.to_datetime(df["timestamp_brussels"]).dt.hour
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, mode: str = "training") -> pd.DataFrame:
    """Main feature-engineering entry point.

    Applies lag, rolling, cyclical, and interaction transforms in order.

    Args:
        df: DataFrame with columns matching the gold feature-base schema.
        mode: ``"training"`` uses weather actuals; ``"serving"`` expects
            forecast columns (same names). Both produce identical output.

    Returns:
        DataFrame with all engineered features appended.
    """
    if df.empty:
        return df
    df = compute_lag_features(df)
    df = compute_rolling_features(df)
    df = compute_cyclical_features(df)
    df = compute_interaction_features(df)
    return df


# ---------------------------------------------------------------------------
# Feature column list (from config)
# ---------------------------------------------------------------------------

def get_feature_columns() -> list[str]:
    """Return the canonical ordered list of model input feature names.

    The list is read from ``configs/data/features.yaml`` (key
    ``feature_columns``), ensuring a single source of truth.

    Returns:
        Ordered list of feature column name strings.
    """
    cfg = _load_features_config()
    return list(cfg["feature_columns"])
