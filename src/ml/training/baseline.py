"""
Baseline models for Belgian energy demand forecasting.

Two baselines:
    1. Persistence: predict = load_lag_24h (yesterday same hour)
    2. Linear Regression: sklearn LinearRegression on the same feature set

Both are logged to MLflow in the same experiment as the main model for
easy comparison.

Usage:
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
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ml.features.feature_engineering import get_feature_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_split_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "training" / "lightgbm.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("split", {})
    return {"train_end": "2024-06-30", "val_end": "2024-09-30"}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        if mask.sum() > 0
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_baselines() -> None:
    """Train and log both baseline models."""
    # --- Load data ---
    data_path = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    df = pd.read_parquet(data_path)
    df = df.sort_values("timestamp_brussels").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp_brussels"])

    split_cfg = _load_split_config()
    train_end = pd.Timestamp(split_cfg["train_end"])
    val_end = pd.Timestamp(split_cfg["val_end"])

    train_mask = ts <= train_end
    test_mask = ts > val_end

    target_col = "target_load_24h"
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    y_test = df.loc[test_mask, target_col].values

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("energy-demand-forecast")

    # ------------------------------------------------------------------
    # Baseline 1: Persistence (predict = load_lag_24h)
    # ------------------------------------------------------------------
    logger.info("Running persistence baseline...")
    with mlflow.start_run(run_name="baseline-persistence"):
        mlflow.set_tag("model_type", "baseline-persistence")
        mlflow.set_tag("python_version", platform.python_version())

        if "load_lag_24h" not in df.columns:
            logger.error(
                "Column 'load_lag_24h' not found. Cannot run persistence baseline."
            )
        else:
            y_pred_persist = df.loc[test_mask, "load_lag_24h"].values
            # Drop rows where lag is NaN
            valid = ~np.isnan(y_pred_persist) & ~np.isnan(y_test)
            metrics = _compute_metrics(y_test[valid], y_pred_persist[valid])
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.log_params({"method": "persistence", "lag_hours": 24})
            logger.info("Persistence baseline test metrics: %s", metrics)

    # ------------------------------------------------------------------
    # Baseline 2: Linear Regression
    # ------------------------------------------------------------------
    logger.info("Running linear regression baseline...")
    with mlflow.start_run(run_name="baseline-linear-regression"):
        mlflow.set_tag("model_type", "baseline-linear-regression")
        mlflow.set_tag("python_version", platform.python_version())

        X_train = df.loc[train_mask, feature_cols].copy()
        y_train = df.loc[train_mask, target_col].values
        X_test_lr = df.loc[test_mask, feature_cols].copy()

        # Fill NaN for linear regression
        X_train = X_train.fillna(0)
        X_test_lr = X_test_lr.fillna(0)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test_lr)

        metrics = _compute_metrics(y_test, y_pred_lr)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_params(
            {"method": "linear_regression", "n_features": len(feature_cols)}
        )
        mlflow.sklearn.log_model(lr, artifact_path="model")
        logger.info("Linear regression baseline test metrics: %s", metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_baselines()
    print("Baseline models logged to MLflow.")


if __name__ == "__main__":
    main()
