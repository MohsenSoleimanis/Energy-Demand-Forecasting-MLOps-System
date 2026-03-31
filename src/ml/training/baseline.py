"""Baseline models for Belgian energy demand forecasting.

Two baselines are trained and logged to MLflow for comparison with the
main LightGBM model:

    1. **Persistence** -- predict = ``load_lag_24h`` (same hour yesterday).
    2. **Linear Regression** -- scikit-learn ``LinearRegression`` on the
       full feature set.

Usage::

    python -m src.ml.training.baseline
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.ml.features.engineering import get_feature_columns
from src.ml.training.data import (
    get_feature_target_split,
    load_training_data,
    temporal_split,
)
from src.shared.config import load_config
from src.shared.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "lightgbm.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"


# ---------------------------------------------------------------------------
# Baseline trainers
# ---------------------------------------------------------------------------

def train_persistence_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: dict,
) -> dict[str, float]:
    """Evaluate the persistence baseline (lag-24h) on the test set.

    Args:
        df: Full time-sorted DataFrame.
        feature_cols: Feature column names (unused here but kept for API
            symmetry).
        target_col: Target column name.
        config: Training configuration dict.

    Returns:
        Dictionary of test metrics.
    """
    _, _, test_df, _ = temporal_split(
        df, config["split"]["train_ratio"], config["split"]["val_ratio"],
    )

    if "load_lag_24h" not in df.columns:
        logger.error("Column 'load_lag_24h' not found. Cannot run persistence baseline.")
        return {}

    y_test = test_df[target_col].values
    y_pred = test_df["load_lag_24h"].values

    valid = ~np.isnan(y_pred) & ~np.isnan(y_test)
    metrics = compute_regression_metrics(y_test[valid], y_pred[valid])
    return metrics


def train_linear_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: dict,
) -> dict[str, float]:
    """Train a linear regression baseline on the same split.

    Args:
        df: Full time-sorted DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        config: Training configuration dict.

    Returns:
        Dictionary of test metrics.
    """
    train_df, _, test_df, _ = temporal_split(
        df, config["split"]["train_ratio"], config["split"]["val_ratio"],
    )

    X_train, y_train = get_feature_target_split(train_df, feature_cols, target_col)
    X_test, y_test = get_feature_target_split(test_df, feature_cols, target_col)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    metrics = compute_regression_metrics(y_test.values, y_pred)
    return metrics


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_baselines() -> None:
    """Train both baselines and log results to MLflow.

    Reads all configuration from ``configs/training/lightgbm.yaml``
    (SSoT) and reuses the same temporal split as the main model.
    """
    cfg = load_config(CONFIG_PATH)

    df = load_training_data(DATA_PATH)

    target_col = "target_load_24h"
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg["experiment_name"])

    # -- persistence --
    logger.info("Running persistence baseline...")
    with mlflow.start_run(run_name="baseline-persistence"):
        mlflow.set_tag("model_type", "baseline-persistence")
        mlflow.set_tag("python_version", platform.python_version())

        metrics = train_persistence_baseline(df, feature_cols, target_col, cfg)
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.log_params({"method": "persistence", "lag_hours": 24})
            logger.info("Persistence baseline test metrics: %s", metrics)

    # -- linear regression --
    logger.info("Running linear regression baseline...")
    with mlflow.start_run(run_name="baseline-linear-regression"):
        mlflow.set_tag("model_type", "baseline-linear-regression")
        mlflow.set_tag("python_version", platform.python_version())

        metrics = train_linear_baseline(df, feature_cols, target_col, cfg)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_params({"method": "linear_regression", "n_features": len(feature_cols)})
        logger.info("Linear regression baseline test metrics: %s", metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m src.ml.training.baseline``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_baselines()
    print("Baseline models logged to MLflow.")


if __name__ == "__main__":
    main()
