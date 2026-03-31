"""Data loading and temporal splitting for model training.

Responsible for reading the gold-layer parquet file and producing
temporally ordered train / validation / test splits.  No model logic,
no evaluation, no plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_training_data(path: Path) -> pd.DataFrame:
    """Load and sort the gold training-set parquet file.

    Args:
        path: Absolute path to the parquet file.

    Returns:
        DataFrame sorted by ``timestamp_brussels`` with the column
        cast to ``datetime64``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the required timestamp column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Run `python -m src.ml.training.pull_gold` first."
        )
    df = pd.read_parquet(path)
    if "timestamp_brussels" not in df.columns:
        raise ValueError("Column 'timestamp_brussels' not found in training data.")
    df = df.sort_values("timestamp_brussels").reset_index(drop=True)
    df["timestamp_brussels"] = pd.to_datetime(df["timestamp_brussels"])
    logger.info("Loaded training data: %d rows, %d columns", len(df), len(df.columns))
    return df


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Split a time-sorted DataFrame into train / val / test sets.

    The split is purely index-based using the provided ratios.  The
    remaining fraction ``1 - train_ratio - val_ratio`` becomes the test
    set.

    Args:
        df: Time-sorted DataFrame (must contain ``timestamp_brussels``).
        train_ratio: Fraction of rows for training (e.g. 0.70).
        val_ratio: Fraction of rows for validation (e.g. 0.15).

    Returns:
        ``(train_df, val_df, test_df, boundaries)`` where *boundaries*
        maps ``train_end`` and ``val_end`` to their ``Timestamp`` values
        for reproducibility logging.
    """
    n = len(df)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:]

    boundaries = {
        "train_end": train_df["timestamp_brussels"].iloc[-1],
        "val_end": val_df["timestamp_brussels"].iloc[-1],
    }
    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Boundaries: train_end=%s, val_end=%s",
        boundaries["train_end"], boundaries["val_end"],
    )
    return train_df, val_df, test_df, boundaries


def walk_forward_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 8000,
    val_size: int = 720,
    gap_hours: int = 24,
) -> list[dict]:
    """Generate walk-forward cross-validation splits for time series.

    Each fold has an expanding training window and a fixed-size validation
    window. A gap between train and val prevents information leakage from
    lag features (our model uses 24h lags, so gap must be >= 24h).

    Args:
        df: Sorted DataFrame with ``timestamp_brussels``.
        n_splits: Number of CV folds.
        min_train_size: Minimum training set size (rows).
        val_size: Validation window size (rows).
        gap_hours: Gap between train end and val start (rows, hourly data).

    Returns:
        List of dicts with ``train_idx``, ``val_idx``, ``train_end``,
        ``val_start``, ``val_end``.

    Raises:
        ValueError: If the dataset is too small for the requested splits.
    """
    n = len(df)
    min_required = min_train_size + gap_hours + val_size
    if n < min_required:
        raise ValueError(
            f"Dataset has {n} rows but needs at least {min_required} "
            f"(min_train_size={min_train_size} + gap={gap_hours} + "
            f"val_size={val_size})."
        )

    usable = n - min_train_size - gap_hours - val_size
    if usable < 0:
        raise ValueError(
            f"Not enough data for even one split. "
            f"Need {min_required} rows, have {n}."
        )

    step = max(usable // max(n_splits - 1, 1), 1) if n_splits > 1 else 0
    splits: list[dict] = []

    for i in range(n_splits):
        split_point = min_train_size + i * step
        val_start_idx = split_point + gap_hours
        val_end_idx = val_start_idx + val_size

        if val_end_idx > n:
            logger.warning(
                "Fold %d would exceed dataset length (%d > %d), stopping.",
                i, val_end_idx, n,
            )
            break

        train_idx = list(range(0, split_point))
        val_idx = list(range(val_start_idx, val_end_idx))

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_end": df["timestamp_brussels"].iloc[split_point - 1],
            "val_start": df["timestamp_brussels"].iloc[val_start_idx],
            "val_end": df["timestamp_brussels"].iloc[val_end_idx - 1],
        })

    logger.info(
        "Generated %d walk-forward CV splits (min_train=%d, val=%d, gap=%d)",
        len(splits), min_train_size, val_size, gap_hours,
    )
    return splits


def get_feature_target_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target vector from a DataFrame.

    Args:
        df: Source DataFrame.
        feature_cols: Column names to use as features.
        target_col: Column name for the target variable.

    Returns:
        ``(X, y)`` tuple.
    """
    return df[feature_cols], df[target_col]
