"""Feature engineering and data processing for energy demand forecasting.

Provides a configurable :class:`DataProcessor` that creates lag features,
rolling statistics, cyclical time encodings, and degree-day features from raw
energy-demand DataFrames, then splits them into train / validation / test sets
using a strict time-based strategy (no data leakage).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from energy_forecast.data.schemas import DataSplit

# Default configuration values
_DEFAULT_CONFIG: dict[str, Any] = {
    "target_column": "energy_demand_kwh",
    "lag_features": [1, 2, 3, 6, 12, 24, 48, 168],
    "rolling_windows": [24, 168],
    "rolling_stats": ["mean", "std", "min", "max"],
    "cyclical_features": True,
    "degree_day_base_temp": 18.0,  # degC
    "drop_na": True,
    "timestamp_column": "timestamp",
    "building_id_column": "building_id",
}


class DataProcessor:
    """Feature engineering pipeline for energy demand time-series.

    Parameters
    ----------
    config : dict, optional
        Processing configuration.  Missing keys fall back to sensible
        defaults.  Supported keys:

        * ``target_column`` – name of the demand column (default
          ``"energy_demand_kwh"``).
        * ``lag_features`` – list of integer lag periods in hours.
        * ``rolling_windows`` – list of integer window sizes in hours.
        * ``rolling_stats`` – list of aggregation names (``"mean"``,
          ``"std"``, ``"min"``, ``"max"``).
        * ``cyclical_features`` – whether to add sin/cos time encodings.
        * ``degree_day_base_temp`` – base temperature for HDD / CDD.
        * ``drop_na`` – whether to drop rows containing NaN after feature
          creation.
        * ``timestamp_column`` – name of the timestamp column.
        * ``building_id_column`` – name of the building-id column.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = {**_DEFAULT_CONFIG, **(config or {})}
        self._feature_columns: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from a raw energy-demand DataFrame.

        The input DataFrame is **not** mutated.  A copy is returned with all
        new columns appended.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least: ``timestamp``, ``building_id``,
            ``energy_demand_kwh``, ``temperature``, ``humidity``.

        Returns
        -------
        pd.DataFrame
            Original columns plus engineered features.
        """
        ts_col = self.config["timestamp_column"]
        bid_col = self.config["building_id_column"]
        target = self.config["target_column"]

        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col])
        df.sort_values([bid_col, ts_col], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- Lag features ---
        df = self._add_lag_features(df, target, bid_col)

        # --- Rolling statistics ---
        df = self._add_rolling_features(df, target, bid_col)

        # --- Cyclical time encodings ---
        if self.config.get("cyclical_features", True):
            df = self._add_cyclical_features(df, ts_col)

        # --- Boolean features ---
        df = self._add_boolean_features(df, ts_col)

        # --- Degree-day features ---
        df = self._add_degree_day_features(df)

        # --- Drop rows with NaN (from lag / rolling computation) ---
        if self.config.get("drop_na", True):
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_column: str | None = None,
    ) -> DataSplit:
        """Time-based train / validation / test split.

        **No shuffling** is performed — the split respects temporal ordering
        so that validation and test data always come after training data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame (output of :meth:`create_features`).
        train_ratio, val_ratio, test_ratio : float
            Fractional sizes.  They are normalised internally so they need not
            sum to exactly 1.0.
        target_column : str, optional
            Override the configured target column name.

        Returns
        -------
        DataSplit
        """
        total = train_ratio + val_ratio + test_ratio
        if total <= 0:
            raise ValueError("Split ratios must be positive and sum to > 0")
        train_ratio /= total
        val_ratio /= total

        target = target_column or self.config["target_column"]
        ts_col = self.config["timestamp_column"]
        bid_col = self.config["building_id_column"]

        # Sort globally by time (across buildings) to guarantee temporal order.
        df = df.sort_values([ts_col, bid_col]).reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        exclude_cols = {ts_col, bid_col, target}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self._feature_columns = feature_cols

        X_train = df.iloc[:train_end][feature_cols].to_numpy(dtype=np.float64)
        y_train = df.iloc[:train_end][target].to_numpy(dtype=np.float64)
        X_val = df.iloc[train_end:val_end][feature_cols].to_numpy(dtype=np.float64)
        y_val = df.iloc[train_end:val_end][target].to_numpy(dtype=np.float64)
        X_test = df.iloc[val_end:][feature_cols].to_numpy(dtype=np.float64)
        y_test = df.iloc[val_end:][target].to_numpy(dtype=np.float64)

        return DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
        )

    @staticmethod
    def normalize(
        X_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        X_test: NDArray[np.float64],
        method: str = "standard",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Any]:
        """Fit a scaler on ``X_train`` and apply to all splits.

        Parameters
        ----------
        X_train, X_val, X_test : ndarray
            Feature matrices.
        method : str
            ``"standard"`` for zero-mean unit-variance, ``"minmax"`` for
            [0, 1] scaling.

        Returns
        -------
        tuple
            ``(X_train_scaled, X_val_scaled, X_test_scaled, scaler)``
        """
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalisation method: {method!r}")

        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s, scaler

    @property
    def feature_columns(self) -> list[str] | None:
        """Feature column names set after the most recent :meth:`split_data` call."""
        return self._feature_columns

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_lag_features(
        self, df: pd.DataFrame, target: str, bid_col: str
    ) -> pd.DataFrame:
        """Add lag features grouped by building."""
        lags: list[int] = self.config.get("lag_features", [])
        grouped = df.groupby(bid_col, sort=False)[target]
        for lag in lags:
            col_name = f"lag_{lag}h"
            df[col_name] = grouped.shift(lag)
        return df

    def _add_rolling_features(
        self, df: pd.DataFrame, target: str, bid_col: str
    ) -> pd.DataFrame:
        """Add rolling window statistics grouped by building."""
        windows: list[int] = self.config.get("rolling_windows", [])
        stats: list[str] = self.config.get("rolling_stats", [])

        for window in windows:
            rolling = df.groupby(bid_col, sort=False)[target].rolling(
                window=window, min_periods=window
            )
            for stat_name in stats:
                col_name = f"rolling_{stat_name}_{window}h"
                if stat_name == "mean":
                    series = rolling.mean()
                elif stat_name == "std":
                    series = rolling.std()
                elif stat_name == "min":
                    series = rolling.min()
                elif stat_name == "max":
                    series = rolling.max()
                else:
                    raise ValueError(f"Unsupported rolling statistic: {stat_name!r}")
                # rolling within groupby returns a MultiIndex; reset to align.
                df[col_name] = series.droplevel(0).values
        return df

    @staticmethod
    def _add_cyclical_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """Encode hour, day-of-week, and month as sin/cos pairs."""
        ts = pd.to_datetime(df[ts_col])
        hour = ts.dt.hour.to_numpy(dtype=np.float64)
        dow = ts.dt.dayofweek.to_numpy(dtype=np.float64)
        month = ts.dt.month.to_numpy(dtype=np.float64)

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * dow / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
        return df

    @staticmethod
    def _add_boolean_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """Add ``is_weekend`` and ``is_business_hour`` flags."""
        ts = pd.to_datetime(df[ts_col])
        dow = ts.dt.dayofweek
        hour = ts.dt.hour

        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_business_hour"] = ((hour >= 8) & (hour < 18) & (dow < 5)).astype(int)
        return df

    def _add_degree_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add heating and cooling degree-day features."""
        base = self.config.get("degree_day_base_temp", 18.0)
        temp = df["temperature"].to_numpy(dtype=np.float64)
        df["heating_degree_days"] = np.maximum(base - temp, 0.0)
        df["cooling_degree_days"] = np.maximum(temp - base, 0.0)
        return df
