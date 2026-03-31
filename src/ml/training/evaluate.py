"""Model evaluation, sliced metrics, and diagnostic plots.

Loads a trained model from MLflow, evaluates it on the held-out test set,
produces sliced metrics (by hour, month, weekend), generates diagnostic
plots, and logs everything back to the MLflow run.

Usage::

    python -m src.ml.training.evaluate [--run-id <MLFLOW_RUN_ID>]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ml.features.feature_engineering import get_feature_columns  # noqa: E402
from src.ml.training.data import (  # noqa: E402
    get_feature_target_split,
    load_training_data,
    temporal_split,
)
from src.shared.config import load_config  # noqa: E402
from src.shared.metrics import compute_regression_metrics  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "lightgbm.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
METRICS_DIR = PROJECT_ROOT / "metrics"
PLOTS_DIR = PROJECT_ROOT / "plots"


# ---------------------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------------------

def load_model_and_data(
    run_id: str,
) -> tuple[object, pd.DataFrame, pd.Series, pd.Series]:
    """Load the MLflow model and reconstruct the test split.

    Args:
        run_id: MLflow run ID that produced the model.

    Returns:
        ``(model, X_test, y_test, timestamps)`` where *timestamps* are
        the ``timestamp_brussels`` values for the test rows.
    """
    cfg = load_config(CONFIG_PATH)

    df = load_training_data(DATA_PATH)
    _, _, test_df, _ = temporal_split(
        df, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"],
    )

    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    target_col = "target_load_24h"
    X_test, y_test = get_feature_target_split(test_df, feature_cols, target_col)
    timestamps = test_df["timestamp_brussels"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    return model, X_test, y_test, timestamps


# ---------------------------------------------------------------------------
# Sliced metrics
# ---------------------------------------------------------------------------

def _sliced_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series,
) -> dict:
    """Compute MAE sliced by hour, month, and weekday/weekend.

    Args:
        y_test: True target values.
        y_pred: Predicted values.
        timestamps: Corresponding timestamps.

    Returns:
        Dictionary with ``mae_by_hour``, ``mae_by_month``,
        ``mae_weekend``, and ``mae_weekday`` keys.
    """
    errors = np.abs(y_test - y_pred)
    hour = timestamps.dt.hour
    month = timestamps.dt.month
    is_weekend = timestamps.dt.dayofweek >= 5

    err_series = pd.Series(errors, index=timestamps.index)
    mae_by_hour = err_series.groupby(hour).mean()
    mae_by_month = err_series.groupby(month).mean()
    mae_weekend = float(errors[is_weekend.values].mean()) if is_weekend.any() else float("nan")
    mae_weekday = float(errors[~is_weekend.values].mean()) if (~is_weekend).any() else float("nan")

    return {
        "mae_by_hour": {int(k): float(v) for k, v in mae_by_hour.items()},
        "mae_by_month": {int(k): float(v) for k, v in mae_by_month.items()},
        "mae_weekend": mae_weekend,
        "mae_weekday": mae_weekday,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    """Save *fig* to *output_dir/name* and close it."""
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series,
    output_dir: Path,
    model: object | None = None,
    X_test: pd.DataFrame | None = None,
    shap_requested: bool = False,
) -> list[Path]:
    """Create all diagnostic plots and return their paths.

    Args:
        y_test: True target values.
        y_pred: Predicted values.
        timestamps: Corresponding timestamps.
        output_dir: Directory to write PNG files into.
        model: Trained model (used for feature importance and SHAP).
        X_test: Test features (used for SHAP).
        shap_requested: If ``True`` and SHAP is not installed,
            raise ``ImportError`` rather than silently skipping.

    Returns:
        List of paths to generated plot files.
    """
    output_dir.mkdir(exist_ok=True)
    paths: list[Path] = []

    paths.append(_plot_actual_vs_predicted(y_test, y_pred, output_dir))
    paths.append(_plot_residuals_over_time(timestamps, y_test - y_pred, output_dir))

    sliced = _sliced_metrics(y_test, y_pred, timestamps)
    paths.append(_plot_error_by_hour(sliced["mae_by_hour"], output_dir))
    paths.append(_plot_error_by_month(sliced["mae_by_month"], output_dir))

    if model is not None:
        paths.append(_plot_feature_importance(model, output_dir))

    if model is not None and X_test is not None:
        shap_path = _plot_shap_summary(model, X_test, output_dir, shap_requested)
        if shap_path is not None:
            paths.append(shap_path)

    return paths


def _plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path,
) -> Path:
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Load (MW)")
    ax.set_ylabel("Predicted Load (MW)")
    ax.set_title("Actual vs Predicted")
    return _save_fig(fig, "actual_vs_predicted.png", output_dir)


def _plot_residuals_over_time(
    timestamps: pd.Series, residuals: np.ndarray, output_dir: Path,
) -> Path:
    """Line plot of residuals over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pd.to_datetime(timestamps), residuals, linewidth=0.3, alpha=0.7)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (MW)")
    ax.set_title("Residuals Over Time")
    return _save_fig(fig, "residuals_over_time.png", output_dir)


def _plot_error_by_hour(mae_by_hour: dict[int, float], output_dir: Path) -> Path:
    """Bar chart of MAE by hour of day."""
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = sorted(mae_by_hour.keys())
    ax.bar(hours, [mae_by_hour[h] for h in hours])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("MAE by Hour of Day")
    ax.set_xticks(range(24))
    return _save_fig(fig, "error_by_hour.png", output_dir)


def _plot_error_by_month(mae_by_month: dict[int, float], output_dir: Path) -> Path:
    """Bar chart of MAE by month."""
    fig, ax = plt.subplots(figsize=(10, 5))
    months = sorted(mae_by_month.keys())
    ax.bar(months, [mae_by_month[m] for m in months])
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("MAE by Month")
    ax.set_xticks(range(1, 13))
    return _save_fig(fig, "error_by_month.png", output_dir)


def _plot_feature_importance(model: object, output_dir: Path) -> Path:
    """LightGBM gain-based feature importance plot."""
    import lightgbm as lgb

    fig, ax = plt.subplots(figsize=(10, 8))
    lgb.plot_importance(model, ax=ax, max_num_features=20, importance_type="gain")
    plt.tight_layout()
    return _save_fig(fig, "feature_importance.png", output_dir)


def _plot_shap_summary(
    model: object,
    X_test: pd.DataFrame,
    output_dir: Path,
    shap_requested: bool,
) -> Path | None:
    """SHAP summary plot (optional dependency).

    Args:
        model: Trained model.
        X_test: Test features.
        output_dir: Plot output directory.
        shap_requested: If ``True`` and SHAP is missing, raise
            ``ImportError`` instead of returning ``None``.

    Returns:
        Path to the saved plot, or ``None`` if SHAP is unavailable.

    Raises:
        ImportError: When *shap_requested* is ``True`` and SHAP is not
            installed.
    """
    try:
        import shap
    except ImportError:
        if shap_requested:
            raise ImportError(
                "SHAP was explicitly requested but is not installed. "
                "Install it with: pip install shap"
            )
        logger.info("SHAP not installed, skipping explainability plot")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.head(500))
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test.head(500), show=False)
        path = _save_fig(fig, "shap_summary.png", output_dir)
        return path
    except Exception as exc:
        logger.warning("SHAP summary plot failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(run_id: str | None = None) -> dict:
    """Evaluate a trained model and log results to MLflow.

    If *run_id* is ``None``, reads it from ``metrics/run_id.txt``.

    Args:
        run_id: MLflow run ID to evaluate.

    Returns:
        Dictionary of global and sliced metrics.
    """
    if run_id is None:
        run_id = _read_run_id()

    model, X_test, y_test, timestamps = load_model_and_data(run_id)
    y_pred = model.predict(X_test)

    # -- global metrics (SSoT: shared utility) --
    metrics: dict = compute_regression_metrics(y_test.values, y_pred)

    # -- sliced metrics --
    sliced = _sliced_metrics(y_test.values, y_pred, timestamps)
    metrics.update(sliced)

    # -- plots --
    artifact_paths = generate_plots(
        y_test.values, y_pred, timestamps, PLOTS_DIR,
        model=model, X_test=X_test,
    )

    # -- persist --
    METRICS_DIR.mkdir(exist_ok=True)
    metrics_path = METRICS_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # -- log to MLflow --
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_id=run_id):
        for k in ("mae", "rmse", "mape", "r2", "mae_weekend", "mae_weekday"):
            mlflow.log_metric(f"test_{k}", metrics[k])
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="plots")
        mlflow.log_artifact(str(metrics_path))

    logger.info("Evaluation complete. Metrics saved to %s", metrics_path)
    return metrics


def _read_run_id() -> str:
    """Read run_id from the DVC pipeline output file.

    Raises:
        FileNotFoundError: If the run_id file does not exist.
    """
    run_id_file = METRICS_DIR / "run_id.txt"
    if not run_id_file.exists():
        raise FileNotFoundError(
            f"run_id file not found at {run_id_file}. "
            "Run training first or pass --run-id explicitly."
        )
    run_id = run_id_file.read_text().strip()
    logger.info("Read run_id from %s: %s", run_id_file, run_id)
    return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m src.ml.training.evaluate``."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--run-id", default=None, help="MLflow run ID to evaluate")
    args = parser.parse_args()

    metrics = run_evaluation(run_id=args.run_id)
    scalar = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    print(json.dumps(scalar, indent=2))


if __name__ == "__main__":
    main()
