"""
Comprehensive model evaluation for Belgian energy demand forecasting.

Generates metrics (MAE, RMSE, MAPE, R2) sliced by hour, month, and weekend,
produces diagnostic plots, and logs everything to MLflow.

Usage:
    python -m src.ml.training.evaluate --run-id <MLFLOW_RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ml.features.feature_engineering import get_feature_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": _mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_and_log(fig: plt.Figure, name: str, plots_dir: Path) -> str:
    path = plots_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, plots_dir: Path
) -> str:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Load (MW)")
    ax.set_ylabel("Predicted Load (MW)")
    ax.set_title("Actual vs Predicted")
    return _save_and_log(fig, "actual_vs_predicted.png", plots_dir)


def plot_error_by_hour(mae_by_hour: pd.Series, plots_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(mae_by_hour.index, mae_by_hour.values)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("MAE by Hour of Day")
    ax.set_xticks(range(24))
    return _save_and_log(fig, "error_by_hour.png", plots_dir)


def plot_error_by_month(mae_by_month: pd.Series, plots_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(mae_by_month.index, mae_by_month.values)
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("MAE by Month")
    ax.set_xticks(range(1, 13))
    return _save_and_log(fig, "error_by_month.png", plots_dir)


def plot_residuals_over_time(
    timestamps: pd.Series, residuals: np.ndarray, plots_dir: Path
) -> str:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps, residuals, linewidth=0.3, alpha=0.7)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (MW)")
    ax.set_title("Residuals Over Time")
    return _save_and_log(fig, "residuals_over_time.png", plots_dir)


def plot_shap_summary(model, X: pd.DataFrame, plots_dir: Path) -> str | None:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.head(500))
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X.head(500), show=False)
        path = _save_and_log(fig, "shap_summary.png", plots_dir)
        return path
    except Exception as exc:
        logger.warning("SHAP summary failed: %s", exc)
        return None


def plot_feature_importance(model, plots_dir: Path) -> str:
    import lightgbm as lgb

    fig, ax = plt.subplots(figsize=(10, 8))
    lgb.plot_importance(model, ax=ax, max_num_features=20, importance_type="gain")
    plt.tight_layout()
    return _save_and_log(fig, "feature_importance.png", plots_dir)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(run_id: str) -> dict:
    """Load model from *run_id*, evaluate on test set, write artifacts."""
    # --- Load data ---
    data_path = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
    df = pd.read_parquet(data_path)
    df = df.sort_values("timestamp_brussels").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp_brussels"])

    # --- Load model ---
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)

    # --- Determine test split from run params ---
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id).data
    val_end = pd.Timestamp(run_data.params.get("val_end", "2024-09-30"))
    test_mask = ts > val_end

    target_col = "target_load_24h"
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col].values
    ts_test = ts[test_mask]

    y_pred = model.predict(X_test)

    # --- Global metrics ---
    metrics = compute_all_metrics(y_test, y_pred)

    # --- Sliced metrics ---
    hour_of_day = ts_test.dt.hour
    month = ts_test.dt.month
    is_weekend = ts_test.dt.dayofweek >= 5

    errors = np.abs(y_test - y_pred)
    mae_by_hour = pd.Series(errors, index=ts_test.index).groupby(hour_of_day).mean()
    mae_by_month = pd.Series(errors, index=ts_test.index).groupby(month).mean()
    mae_weekend = float(errors[is_weekend.values].mean()) if is_weekend.any() else float("nan")
    mae_weekday = float(errors[~is_weekend.values].mean()) if (~is_weekend).any() else float("nan")

    metrics["mae_by_hour"] = {int(k): float(v) for k, v in mae_by_hour.items()}
    metrics["mae_by_month"] = {int(k): float(v) for k, v in mae_by_month.items()}
    metrics["mae_weekend"] = mae_weekend
    metrics["mae_weekday"] = mae_weekday

    # --- Plots ---
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    artifact_paths = []
    artifact_paths.append(plot_actual_vs_predicted(y_test, y_pred, plots_dir))
    artifact_paths.append(plot_error_by_hour(mae_by_hour, plots_dir))
    artifact_paths.append(plot_error_by_month(mae_by_month, plots_dir))
    artifact_paths.append(
        plot_residuals_over_time(pd.to_datetime(ts_test), y_test - y_pred, plots_dir)
    )
    artifact_paths.append(plot_feature_importance(model, plots_dir))

    shap_path = plot_shap_summary(model, X_test, plots_dir)
    if shap_path:
        artifact_paths.append(shap_path)

    # --- Save metrics JSON ---
    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_path = metrics_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Log to MLflow ---
    with mlflow.start_run(run_id=run_id):
        for k in ("mae", "rmse", "mape", "r2", "mae_weekend", "mae_weekday"):
            mlflow.log_metric(f"test_{k}", metrics[k])
        for path in artifact_paths:
            mlflow.log_artifact(path, artifact_path="plots")
        mlflow.log_artifact(str(metrics_path))

    logger.info("Evaluation complete. Metrics saved to %s", metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to evaluate")
    args = parser.parse_args()
    metrics = evaluate(args.run_id)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
