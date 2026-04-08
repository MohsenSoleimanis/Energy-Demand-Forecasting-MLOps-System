"""Data validation for energy demand forecasting.

Implements pure pandas / numpy validation checks that follow Great
Expectations-style patterns without requiring the heavy GE dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from energy_forecast.data.schemas import ValidationResult

# Columns required in the raw data
_RAW_REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "building_id",
    "energy_demand_kwh",
    "temperature",
    "humidity",
    "is_holiday",
    "day_of_week",
]

# Columns that must not contain null values
_RAW_NON_NULL_COLUMNS: list[str] = [
    "timestamp",
    "building_id",
    "energy_demand_kwh",
    "temperature",
]

# Acceptable value ranges for raw columns
_RAW_RANGES: dict[str, tuple[float | None, float | None]] = {
    "energy_demand_kwh": (0.0, None),  # strictly positive, checked separately
    "temperature": (-40.0, 50.0),
    "humidity": (0.0, 100.0),
    "day_of_week": (0, 6),
}


class DataValidator:
    """Validates raw and feature-engineered DataFrames for data quality.

    All public methods return a :class:`ValidationResult` containing lists of
    errors (hard failures), warnings (soft issues), and summary statistics.
    The ``is_valid`` flag is ``True`` only when there are zero errors.

    This class does **not** depend on Great Expectations; all checks use plain
    pandas / numpy so that validation can run in lightweight environments.
    """

    # ------------------------------------------------------------------
    # Raw data validation
    # ------------------------------------------------------------------

    def validate_raw_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a raw energy-demand DataFrame.

        Checks performed:

        1. Required columns are present.
        2. No nulls in key columns.
        3. ``energy_demand_kwh`` > 0.
        4. Temperature in [−40, 50] °C.
        5. Humidity in [0, 100] %.
        6. ``day_of_week`` in [0, 6].
        7. Timestamps are monotonically non-decreasing **per building**.
        8. No duplicate ``(building_id, timestamp)`` pairs.
        """
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, stats=stats
            )

        # 1. Schema check
        missing = set(_RAW_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")
            # Cannot continue further checks if schema is broken.
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, stats=stats
            )

        # Populate basic stats
        stats["num_rows"] = len(df)
        stats["num_buildings"] = int(df["building_id"].nunique())
        stats["date_range"] = (
            str(df["timestamp"].min()),
            str(df["timestamp"].max()),
        )

        # 2. Null checks
        for col in _RAW_NON_NULL_COLUMNS:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                errors.append(f"Column '{col}' contains {n_null} null values")

        # 3–6. Range checks
        self._check_positive(df, "energy_demand_kwh", errors, warnings, stats)
        for col, (lo, hi) in _RAW_RANGES.items():
            if col == "energy_demand_kwh":
                continue  # handled above
            self._check_range(df, col, lo, hi, errors, warnings, stats)

        # 7. Monotonicity of timestamps per building
        self._check_timestamp_monotonicity(df, errors, warnings)

        # 8. Duplicate timestamps per building
        self._check_duplicate_timestamps(df, errors, warnings, stats)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Feature data validation
    # ------------------------------------------------------------------

    def validate_features(
        self,
        df: pd.DataFrame,
        target_column: str = "energy_demand_kwh",
        max_nan_fraction: float = 0.0,
        min_abs_correlation: float = 0.01,
    ) -> ValidationResult:
        """Validate a feature-engineered DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`DataProcessor.create_features`.
        target_column : str
            Name of the target column for correlation checks.
        max_nan_fraction : float
            Maximum allowed fraction of NaN values per column (0.0 means no
            NaNs at all).
        min_abs_correlation : float
            Warn if any numeric feature has an absolute Pearson correlation
            with the target below this threshold.
        """
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        if df.empty:
            errors.append("Feature DataFrame is empty")
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, stats=stats
            )

        stats["num_rows"] = len(df)
        stats["num_columns"] = len(df.columns)

        # 1. NaN check per column
        nan_counts = df.isna().sum()
        nan_fracs = nan_counts / len(df)
        cols_with_nan = nan_fracs[nan_fracs > max_nan_fraction]
        if not cols_with_nan.empty:
            for col, frac in cols_with_nan.items():
                errors.append(
                    f"Column '{col}' has {frac:.2%} NaN values "
                    f"(limit: {max_nan_fraction:.2%})"
                )
        stats["nan_fractions"] = nan_fracs[nan_fracs > 0].to_dict()

        # 2. Numeric feature range checks — flag infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            n_inf = int(np.isinf(df[col]).sum())
            if n_inf > 0:
                errors.append(f"Column '{col}' contains {n_inf} infinite values")

        stats["numeric_columns"] = len(numeric_cols)

        # 3. Feature-target correlation
        if target_column in df.columns:
            numeric_features = [
                c for c in numeric_cols if c != target_column
            ]
            if numeric_features:
                correlations: dict[str, float] = {}
                target_series = df[target_column]
                for col in numeric_features:
                    corr = target_series.corr(df[col])
                    if np.isnan(corr):
                        warnings.append(
                            f"Correlation between '{col}' and target is NaN "
                            "(likely constant column)"
                        )
                        correlations[col] = 0.0
                    else:
                        correlations[col] = round(float(corr), 4)

                low_corr = {
                    k: v
                    for k, v in correlations.items()
                    if abs(v) < min_abs_correlation
                }
                if low_corr:
                    warnings.append(
                        f"{len(low_corr)} features have |correlation| < "
                        f"{min_abs_correlation} with target: "
                        f"{sorted(low_corr.keys())}"
                    )
                stats["target_correlations"] = correlations
        else:
            warnings.append(
                f"Target column '{target_column}' not found — "
                "skipping correlation check"
            )

        # 4. Constant-column check
        constant_cols = [
            col
            for col in numeric_cols
            if df[col].nunique(dropna=True) <= 1
        ]
        if constant_cols:
            warnings.append(f"Constant or single-value columns: {constant_cols}")

        stats["constant_columns"] = constant_cols

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_positive(
        df: pd.DataFrame,
        col: str,
        errors: list[str],
        warnings: list[str],
        stats: dict[str, Any],
    ) -> None:
        """Assert that all values in *col* are strictly positive."""
        if col not in df.columns:
            return
        non_positive = int((df[col] <= 0).sum())
        if non_positive > 0:
            errors.append(
                f"Column '{col}' has {non_positive} non-positive values "
                f"(min={df[col].min():.4f})"
            )
        stats[f"{col}_min"] = float(df[col].min())
        stats[f"{col}_max"] = float(df[col].max())
        stats[f"{col}_mean"] = float(df[col].mean())

    @staticmethod
    def _check_range(
        df: pd.DataFrame,
        col: str,
        lo: float | None,
        hi: float | None,
        errors: list[str],
        warnings: list[str],
        stats: dict[str, Any],
    ) -> None:
        """Assert that values fall within ``[lo, hi]``."""
        if col not in df.columns:
            return
        series = df[col].dropna()
        if series.empty:
            return

        out_of_range = 0
        if lo is not None:
            below = int((series < lo).sum())
            out_of_range += below
        if hi is not None:
            above = int((series > hi).sum())
            out_of_range += above

        if out_of_range > 0:
            errors.append(
                f"Column '{col}' has {out_of_range} values outside "
                f"[{lo}, {hi}] (actual range: [{series.min():.2f}, {series.max():.2f}])"
            )

        stats[f"{col}_min"] = float(series.min())
        stats[f"{col}_max"] = float(series.max())

    @staticmethod
    def _check_timestamp_monotonicity(
        df: pd.DataFrame,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Ensure timestamps are non-decreasing within each building."""
        ts = pd.to_datetime(df["timestamp"])
        bid = df["building_id"]

        for building, group in df.groupby(bid, sort=False):
            ts_group = pd.to_datetime(group["timestamp"])
            diffs = ts_group.diff().dropna()
            n_decreasing = int((diffs < pd.Timedelta(0)).sum())
            if n_decreasing > 0:
                errors.append(
                    f"Building '{building}': timestamps are not monotonically "
                    f"non-decreasing ({n_decreasing} violations)"
                )

    @staticmethod
    def _check_duplicate_timestamps(
        df: pd.DataFrame,
        errors: list[str],
        warnings: list[str],
        stats: dict[str, Any],
    ) -> None:
        """Check for duplicate (building_id, timestamp) pairs."""
        dupes = df.duplicated(subset=["building_id", "timestamp"], keep=False)
        n_dupes = int(dupes.sum())
        stats["duplicate_rows"] = n_dupes
        if n_dupes > 0:
            errors.append(
                f"Found {n_dupes} duplicate (building_id, timestamp) rows"
            )
