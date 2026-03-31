"""LightGBM model building, training, and Optuna hyperparameter tuning.

This module is responsible ONLY for constructing and fitting models.
Data loading lives in ``data.py``; evaluation lives in ``evaluate.py``.

Usage::

    python -m src.ml.training.train [--tune]
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from src.ml.features.feature_engineering import get_feature_columns
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
METRICS_DIR = PROJECT_ROOT / "metrics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Return the current Git commit SHA, or ``'unknown'`` on failure."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _resolve_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the intersection of canonical features and available columns."""
    all_expected = get_feature_columns()
    available = [c for c in all_expected if c in df.columns]
    missing = [c for c in all_expected if c not in df.columns]
    if missing:
        logger.warning("Missing features (excluded): %s", missing)
    logger.info("Using %d features", len(available))
    return available


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> lgb.LGBMRegressor:
    """Construct a LightGBM regressor from *config* without fitting.

    Args:
        config: Full training configuration dict (loaded from YAML).

    Returns:
        An un-fitted ``LGBMRegressor``.
    """
    model_cfg = config["model"]
    return lgb.LGBMRegressor(
        n_estimators=model_cfg["n_estimators"],
        learning_rate=model_cfg["learning_rate"],
        max_depth=model_cfg["max_depth"],
        num_leaves=model_cfg["num_leaves"],
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        min_child_samples=model_cfg["min_child_samples"],
        reg_alpha=model_cfg["reg_alpha"],
        reg_lambda=model_cfg["reg_lambda"],
        random_state=config["random_seed"],
        verbose=-1,
    )


def train_model(
    model: lgb.LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> lgb.LGBMRegressor:
    """Fit *model* with early stopping on the validation set.

    MLflow LightGBM autologging captures per-iteration metrics
    automatically, replacing the need for manual callbacks.

    Args:
        model: Un-fitted ``LGBMRegressor`` (from :func:`build_model`).
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Full training configuration dict.

    Returns:
        The fitted ``LGBMRegressor``.
    """
    early_stopping_rounds = config["model"]["early_stopping_rounds"]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=50),
        ],
    )
    return model


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> dict:
    """Run Optuna hyperparameter search for LightGBM.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Full training configuration dict.

    Returns:
        Dictionary of best hyperparameters found by Optuna.

    Raises:
        ImportError: If Optuna is not installed.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter tuning but is not installed. "
            "Install it with: pip install optuna"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    random_seed = config["random_seed"]
    n_trials = config["tuning"]["n_trials"]
    early_stopping_rounds = config["model"]["early_stopping_rounds"]

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: minimize validation MAPE."""
        params = {
            "n_estimators": config["model"]["n_estimators"],
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
        }
        trial_model = lgb.LGBMRegressor(
            **params, random_state=random_seed, verbose=-1,
        )
        trial_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        val_pred = trial_model.predict(X_val)
        metrics = compute_regression_metrics(y_val.values, val_pred)

        with mlflow.start_run(nested=True, run_name=f"optuna-trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("val_mape", metrics["mape"])

        return metrics["mape"]

    study = optuna.create_study(direction="minimize", study_name="lgbm-hpo")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best trial: %d, MAPE: %.4f", study.best_trial.number, study.best_value)
    logger.info("Best params: %s", study.best_params)
    return study.best_params


def run_training_pipeline(tune: bool = False) -> str:
    """Orchestrate the full training pipeline.

    1. Load config and data.
    2. Split temporally.
    3. Optionally tune hyperparameters.
    4. Train final model.
    5. Log metrics and model to MLflow.
    6. Write ``run_id`` to ``metrics/run_id.txt`` for DVC.

    Args:
        tune: If ``True``, run Optuna before final training.

    Returns:
        The MLflow run ID.
    """
    cfg = load_config(CONFIG_PATH)

    # -- data --
    df = load_training_data(DATA_PATH)
    train_df, val_df, test_df, boundaries = temporal_split(
        df, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"],
    )

    feature_cols = _resolve_feature_cols(df)
    target_col = "target_load_24h"

    X_train, y_train = get_feature_target_split(train_df, feature_cols, target_col)
    X_val, y_val = get_feature_target_split(val_df, feature_cols, target_col)
    X_test, y_test = get_feature_target_split(test_df, feature_cols, target_col)

    # -- MLflow --
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg["experiment_name"])
    mlflow.lightgbm.autolog(log_models=False)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # -- optional tuning --
        if tune:
            logger.info("Running Optuna hyperparameter tuning ...")
            best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, cfg)
            cfg["model"] = {**cfg["model"], **best_params}
            logger.info("Merged tuned params: %s", best_params)
            mlflow.set_tag("tuned", "True")
        else:
            mlflow.set_tag("tuned", "False")

        # -- build and train --
        model = build_model(cfg)
        model = train_model(model, X_train, y_train, X_val, y_val, cfg)

        # -- metrics --
        val_metrics = compute_regression_metrics(y_val.values, model.predict(X_val))
        test_metrics = compute_regression_metrics(y_test.values, model.predict(X_test))

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        logger.info("Validation metrics: %s", val_metrics)
        logger.info("Test metrics: %s", test_metrics)

        # -- log model artifact --
        mlflow.lightgbm.log_model(model, artifact_path="model")

        # -- log split boundaries and metadata --
        mlflow.log_params({
            "train_end": str(boundaries["train_end"]),
            "val_end": str(boundaries["val_end"]),
            "n_features": len(feature_cols),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "random_seed": cfg["random_seed"],
        })
        mlflow.set_tag("git_commit_sha", _git_sha())
        mlflow.set_tag("python_version", platform.python_version())

        # -- save for DVC pipeline --
        METRICS_DIR.mkdir(exist_ok=True)
        _write_metrics_json(run_id, X_train, val_metrics, test_metrics)
        (METRICS_DIR / "run_id.txt").write_text(run_id)

        logger.info("Training complete. MLflow run_id=%s", run_id)
        return run_id


def _write_metrics_json(
    run_id: str,
    X_train: pd.DataFrame,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    """Persist metrics JSON for the DVC pipeline."""
    payload = {
        "run_id": run_id,
        "training_rows": len(X_train),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    with open(METRICS_DIR / "train_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m src.ml.training.train``."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM energy demand model")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning before final training",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_id = run_training_pipeline(tune=args.tune)
    print(f"Training finished. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
