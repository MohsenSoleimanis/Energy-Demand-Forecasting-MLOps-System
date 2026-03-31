"""
Hyperparameter sweep for LightGBM energy demand forecasting model.

Runs multiple configurations, each logged as a separate MLflow run tagged
with "sweep", then prints a comparison table.

Usage:
    python scripts/sweep.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

import logging
import platform
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is on sys.path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.features.feature_engineering import get_feature_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

SWEEP_CONFIGS: list[dict] = [
    {
        "name": "default",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    },
    {
        "name": "fast-learner",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    },
    {
        "name": "deep-model",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 12,
            "num_leaves": 127,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    },
    {
        "name": "conservative",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    },
    {
        "name": "wide",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 127,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    },
]

# ---------------------------------------------------------------------------
# Metrics helper (same as train.py)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R-squared."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = float("nan")
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_split() -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    list[str],
]:
    """Load training data and perform the same temporal split as train.py."""
    data_path = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run the data pipeline first."
        )

    df = pd.read_parquet(data_path)
    logger.info("Loaded training data: %d rows, %d columns", len(df), len(df.columns))

    if "timestamp_brussels" not in df.columns:
        raise ValueError("Column 'timestamp_brussels' not found in training data.")

    df = df.sort_values("timestamp_brussels").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp_brussels"])

    # Same split boundaries as train.py / lightgbm.yaml defaults
    train_end = pd.Timestamp("2024-06-30")
    val_end = pd.Timestamp("2025-03-31")

    train_mask = ts <= train_end
    val_mask = (ts > train_end) & (ts <= val_end)
    test_mask = ts > val_end

    target_col = "target_load_24h"
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    missing = [c for c in get_feature_columns() if c not in df.columns]
    if missing:
        logger.warning("Missing features (excluded): %s", missing)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, target_col]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single_config(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    early_stopping_rounds: int = 50,
    random_seed: int = 42,
) -> dict:
    """Train one LightGBM config inside an MLflow run. Returns summary dict."""
    name = config["name"]
    params = config["params"]

    with mlflow.start_run(run_name=f"sweep-{name}") as run:
        run_id = run.info.run_id

        # Tags
        mlflow.set_tag("sweep", "true")
        mlflow.set_tag("sweep_config", name)
        mlflow.set_tag("python_version", platform.python_version())

        # Log hyperparams
        mlflow.log_params(
            {
                **params,
                "random_seed": random_seed,
                "early_stopping_rounds": early_stopping_rounds,
                "n_features": len(feature_cols),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
            }
        )

        # Build model
        model = lgb.LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_samples=params["min_child_samples"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=random_seed,
            verbose=-1,
        )

        # MLflow per-iteration callback (same pattern as train.py)
        class MlflowLogCallback:
            """Log validation loss at each boosting round to MLflow."""
            def __init__(self):
                self.order = 25

            def __call__(self, env):
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    mlflow.log_metric(
                        f"{data_name}_{eval_name}",
                        result,
                        step=env.iteration,
                    )

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
            MlflowLogCallback(),
        ]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["val"],
            callbacks=callbacks,
        )

        # Final metrics
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        val_metrics = compute_metrics(y_val.values, val_preds)
        test_metrics = compute_metrics(y_test.values, test_preds)

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Log the best iteration
        best_iter = model.best_iteration_ if model.best_iteration_ else params["n_estimators"]
        mlflow.log_metric("best_iteration", best_iter)

        # Log model artifact
        mlflow.lightgbm.log_model(model, artifact_path="model")

        logger.info("[%s] val_mae=%.2f  test_mae=%.2f  run_id=%s", name, val_metrics["mae"], test_metrics["mae"], run_id)

    return {
        "config": name,
        "run_id": run_id,
        "best_iteration": best_iter,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_mape": val_metrics["mape"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_mape": test_metrics["mape"],
        "test_r2": test_metrics["r2"],
    }


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted table comparing all sweep runs."""
    df = pd.DataFrame(results)

    # Column display order
    cols = [
        "config", "run_id", "best_iteration",
        "val_mae", "val_rmse", "val_mape", "val_r2",
        "test_mae", "test_rmse", "test_mape", "test_r2",
    ]
    df = df[[c for c in cols if c in df.columns]]

    # Shorten run_id for display
    df["run_id"] = df["run_id"].str[:8]

    # Format floats
    float_cols = [c for c in df.columns if c not in ("config", "run_id", "best_iteration")]
    for c in float_cols:
        df[c] = df[c].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "NaN")

    print("\n" + "=" * 120)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)

    # Highlight best config by test MAE
    best_idx = pd.to_numeric(df["test_mae"], errors="coerce").idxmin()
    print(f"\nBest config by test MAE: {df.loc[best_idx, 'config']} "
          f"(test_mae={df.loc[best_idx, 'test_mae']})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting hyperparameter sweep with %d configurations", len(SWEEP_CONFIGS))

    # Load data once
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_split()

    # MLflow setup
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("energy-demand-forecast")

    # Run each configuration
    results: list[dict] = []
    for i, config in enumerate(SWEEP_CONFIGS, 1):
        logger.info(
            "--- Running config %d/%d: %s ---", i, len(SWEEP_CONFIGS), config["name"]
        )
        result = run_single_config(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
        )
        results.append(result)

    # Print comparison
    print_comparison_table(results)
    logger.info("Sweep complete. %d runs logged to MLflow.", len(results))


if __name__ == "__main__":
    main()
