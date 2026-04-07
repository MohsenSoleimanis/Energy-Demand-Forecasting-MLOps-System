"""Advanced feature engineering for energy demand forecasting."""

from __future__ import annotations

from typing import Any, Optional

import holidays
import numpy as np
import pandas as pd

from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)

# Columns that are never treated as features
_META_COLUMNS = {"timestamp", "datetime", "date", "index"}
_TARGET_COLUMNS = {"energy_demand", "demand", "target", "load", "consumption"}


class FeatureEngineer:
    """Create and manage features for energy demand forecasting models.

    The engineer tracks which feature columns it has created so that
    downstream consumers (training, serving) can retrieve the canonical
    feature list via :meth:`get_feature_names`.
    """

    def __init__(self) -> None:
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time encodings derived from a datetime index or column.

        Creates sin/cos encodings for hour-of-day, day-of-week, day-of-year,
        and month so that models can learn periodic patterns without discontinuities.

        Parameters
        ----------
        df:
            DataFrame with a ``DatetimeIndex`` or a column parseable as datetime
            (``timestamp`` or ``datetime``).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with additional time feature columns.
        """
        df = df.copy()
        dt = self._resolve_datetime(df)

        # Cyclical hour encoding (period = 24)
        df["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)

        # Cyclical day-of-week encoding (period = 7)
        df["day_of_week_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # Cyclical month encoding (period = 12)
        df["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

        # Cyclical day-of-year encoding (period = 365)
        df["day_of_year_sin"] = np.sin(2 * np.pi * dt.dayofyear / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * dt.dayofyear / 365)

        logger.info("time_features_created", count=8)
        return df

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "energy_demand",
        lags: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Create lag features for the target variable.

        Parameters
        ----------
        df:
            Input DataFrame.
        target_col:
            Name of the column to lag.
        lags:
            List of lag periods (in number of rows). Defaults to
            ``[1, 2, 3, 6, 12, 24, 48, 168]`` covering short-term through
            one-week lags for hourly data.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with lag columns named ``{target_col}_lag_{k}``.
        """
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24, 48, 168]

        df = df.copy()
        for lag in lags:
            col_name = f"{target_col}_lag_{lag}"
            df[col_name] = df[target_col].shift(lag)

        logger.info("lag_features_created", target=target_col, lags=lags)
        return df

    # ------------------------------------------------------------------
    # Rolling / window features
    # ------------------------------------------------------------------

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = "energy_demand",
        windows: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Create rolling-window statistics for the target variable.

        For each window size the following aggregations are computed:
        mean, standard deviation, min, and max.

        Parameters
        ----------
        df:
            Input DataFrame.
        target_col:
            Column on which to compute rolling stats.
        windows:
            List of window sizes (row count). Defaults to ``[3, 6, 12, 24, 168]``.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with rolling feature columns.
        """
        if windows is None:
            windows = [3, 6, 12, 24, 168]

        df = df.copy()
        for w in windows:
            rolling = df[target_col].rolling(window=w, min_periods=1)
            df[f"{target_col}_rolling_mean_{w}"] = rolling.mean()
            df[f"{target_col}_rolling_std_{w}"] = rolling.std()
            df[f"{target_col}_rolling_min_{w}"] = rolling.min()
            df[f"{target_col}_rolling_max_{w}"] = rolling.max()

        logger.info("rolling_features_created", target=target_col, windows=windows)
        return df

    # ------------------------------------------------------------------
    # Weather-derived features
    # ------------------------------------------------------------------

    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive weather interaction features.

        Expects the DataFrame to contain at least a ``temperature`` column.
        ``humidity`` and ``wind_speed`` are used when present.

        New columns:
        - ``heating_degree_days``: max(0, 18 - temperature)
        - ``cooling_degree_days``: max(0, temperature - 24)
        - ``temp_humidity_interaction``: temperature * humidity (if available)
        - ``wind_chill``: simplified wind chill factor (if wind_speed present)

        Parameters
        ----------
        df:
            Input DataFrame with weather columns.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with weather-derived columns.
        """
        df = df.copy()

        if "temperature" not in df.columns:
            logger.warning("weather_features_skipped", reason="no temperature column")
            return df

        temp = df["temperature"]
        df["heating_degree_days"] = np.maximum(0.0, 18.0 - temp)
        df["cooling_degree_days"] = np.maximum(0.0, temp - 24.0)

        if "humidity" in df.columns:
            df["temp_humidity_interaction"] = temp * df["humidity"]

        if "wind_speed" in df.columns:
            # Simplified wind chill (valid for temperatures below ~10 C)
            ws = df["wind_speed"]
            df["wind_chill"] = np.where(
                temp < 10,
                13.12
                + 0.6215 * temp
                - 11.37 * np.power(ws.clip(lower=0.1), 0.16)
                + 0.3965 * temp * np.power(ws.clip(lower=0.1), 0.16),
                temp,
            )

        logger.info("weather_features_created")
        return df

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    def create_calendar_features(
        self,
        df: pd.DataFrame,
        country: str = "US",
    ) -> pd.DataFrame:
        """Add calendar-based binary and categorical features.

        Parameters
        ----------
        df:
            Input DataFrame with a resolvable datetime.
        country:
            ISO country code used by the ``holidays`` library to determine
            public holidays.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with calendar feature columns.
        """
        df = df.copy()
        dt = self._resolve_datetime(df)

        df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(np.int8)
        df["is_business_hour"] = ((dt.hour >= 8) & (dt.hour < 18)).astype(np.int8)

        # Season: 0=Winter, 1=Spring, 2=Summer, 3=Autumn (northern hemisphere)
        month = dt.month
        df["season"] = np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            [0, 1, 2, 3],
            default=0,
        )

        # Public holidays
        years = dt.year.unique().tolist()
        holiday_dates = set()
        for year in years:
            holiday_dates.update(holidays.country_holidays(country, years=year).keys())

        df["is_holiday"] = dt.date.isin(holiday_dates).astype(np.int8)

        logger.info("calendar_features_created", country=country)
        return df

    # ------------------------------------------------------------------
    # Convenience: create all features at once
    # ------------------------------------------------------------------

    def create_all_features(
        self,
        df: pd.DataFrame,
        config: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Apply all feature-engineering steps in sequence.

        Parameters
        ----------
        df:
            Raw input DataFrame.
        config:
            Optional configuration dict with keys:

            - ``target_col`` (str): target column name, default ``"energy_demand"``
            - ``lags`` (list[int]): lag periods
            - ``windows`` (list[int]): rolling window sizes
            - ``country`` (str): ISO country code for holidays
            - ``drop_na`` (bool): whether to drop rows with NaN from lags/rolling,
              default ``True``

        Returns
        -------
        pd.DataFrame
            Fully-featured DataFrame.
        """
        config = config or {}
        target_col: str = config.get("target_col", "energy_demand")
        lags: Optional[list[int]] = config.get("lags")
        windows: Optional[list[int]] = config.get("windows")
        country: str = config.get("country", "US")
        drop_na: bool = config.get("drop_na", True)

        df = self.create_time_features(df)
        df = self.create_calendar_features(df, country=country)
        df = self.create_weather_features(df)
        df = self.create_lag_features(df, target_col=target_col, lags=lags)
        df = self.create_rolling_features(df, target_col=target_col, windows=windows)

        if drop_na:
            before = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.info("dropped_na_rows", removed=before - len(df), remaining=len(df))

        # Cache the feature column names
        self._feature_names = self._extract_feature_names(df, target_col)

        logger.info("all_features_created", total_features=len(self._feature_names))
        return df

    # ------------------------------------------------------------------
    # Feature name introspection
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names from the last ``create_all_features`` call.

        Excludes target and metadata columns.

        Returns
        -------
        list[str]
            Sorted list of feature column names.
        """
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_datetime(df: pd.DataFrame) -> pd.Series:
        """Return a ``pd.Series`` of datetime accessor from the DataFrame.

        Checks the index first, then looks for common datetime column names.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index

        for col in ("timestamp", "datetime", "date"):
            if col in df.columns:
                return pd.to_datetime(df[col]).dt

        raise ValueError(
            "Cannot resolve datetime. Provide a DatetimeIndex or a column named "
            "'timestamp', 'datetime', or 'date'."
        )

    @staticmethod
    def _extract_feature_names(df: pd.DataFrame, target_col: str) -> list[str]:
        """Identify feature columns by excluding target and metadata columns."""
        exclude = _META_COLUMNS | _TARGET_COLUMNS | {target_col}
        return sorted(col for col in df.columns if col not in exclude)
