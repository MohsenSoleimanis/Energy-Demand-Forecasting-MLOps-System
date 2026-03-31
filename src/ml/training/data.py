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
