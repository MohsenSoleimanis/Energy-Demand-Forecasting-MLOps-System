"""Model building, training, cross-validation, and Optuna hyperparameter tuning.

Supports LightGBM (default), XGBoost, Ridge, and weighted ensemble via
the ``--model-type`` flag.  Walk-forward cross-validation is available
with ``--cv``.  Data loading lives in ``data.py``; evaluation lives in
``evaluate.py``.

Usage::

    python -m src.ml.training.train [--tune] [--cv] [--model-type lightgbm]
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.ml.features.engineering import get_feature_columns
from src.ml.training.data import (
    get_feature_target_split,
    load_training_data,
    temporal_split,
    walk_forward_cv,
)
from src.ml.training.models import (
    WeightedEnsemble,
    build_ensemble,
    get_model_builder,
)
from src.shared.business_metrics import compute_imbalance_cost
from src.shared.config import load_config
from src.shared.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "lightgbm.yaml"
ENSEMBLE_CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "ensemble.yaml"
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

def build_model(config: dict, model_type: str = "lightgbm") -> Any:
    """Construct an un-fitted model from *config*.

    Args:
        config: Full training configuration dict (loaded from YAML).
        model_type: One of ``lightgbm``, ``xgboost``, ``ridge``.

    Returns:
        An un-fitted sklearn-compatible regressor.
    """
    if model_type == "lightgbm":
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
    builder = get_model_builder(model_type)
    model_cfg = config.get("model", {})
    return builder(model_cfg, random_seed=config["random_seed"])


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> Any:
    """Fit *model* with early stopping on the validation set.

    For LightGBM, uses native early stopping callbacks.  For other
    sklearn-compatible models (XGBoost, Ridge), falls back to a plain
    ``.fit()`` call.

    Args:
        model: Un-fitted estimator (from :func:`build_model`).
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Full training configuration dict.

    Returns:
        The fitted model.
    """
    if isinstance(model, lgb.LGBMRegressor):
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
    else:
        # XGBoost and Ridge use plain fit
        model.fit(X_train, y_train)
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
        ) from None

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


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: dict,
    model_type: str = "lightgbm",
) -> dict[str, Any]:
    """Run walk-forward cross-validation and return aggregated metrics.

    Trains a fresh model on each fold's expanding training window and
    evaluates on the fixed validation window.  Logs per-fold metrics as
    nested MLflow runs.

    Args:
        df: Full sorted DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        config: Full training configuration dict.
        model_type: Model architecture to use per fold.

    Returns:
        Dictionary with per-fold metrics, mean/std MAPE, and a
        ``cv_passed`` boolean from the stability quality gate.

    Raises:
        ValueError: If the dataset is too small for the requested splits.
    """
    ensemble_cfg = load_config(ENSEMBLE_CONFIG_PATH)
    cv_cfg = ensemble_cfg.get("cross_validation", {})

    n_splits = cv_cfg.get("n_splits", 5)
    min_train_size = cv_cfg.get("min_train_size", 8000)
    val_size = cv_cfg.get("val_size", 720)
    gap_hours = cv_cfg.get("gap_hours", 24)
    max_mape_std = cv_cfg.get("max_mape_std", 0.02)

    splits = walk_forward_cv(
        df,
        n_splits=n_splits,
        min_train_size=min_train_size,
        val_size=val_size,
        gap_hours=gap_hours,
    )

    fold_metrics: list[dict[str, float]] = []
    fold_mapes: list[float] = []

    for i, split in enumerate(splits):
        logger.info(
            "CV fold %d/%d: train=%d rows, val=%d rows, "
            "train_end=%s, val_start=%s",
            i + 1, len(splits),
            len(split["train_idx"]), len(split["val_idx"]),
            split["train_end"], split["val_start"],
        )

        train_fold = df.iloc[split["train_idx"]]
        val_fold = df.iloc[split["val_idx"]]

        X_tr, y_tr = get_feature_target_split(train_fold, feature_cols, target_col)
        X_va, y_va = get_feature_target_split(val_fold, feature_cols, target_col)

        model = build_model(config, model_type=model_type)
        model = train_model(model, X_tr, y_tr, X_va, y_va, config)

        y_pred = model.predict(X_va)
        metrics = compute_regression_metrics(y_va.values, y_pred)
        biz = compute_imbalance_cost(y_va.values, y_pred)
        metrics.update(biz)

        # Log per-fold metrics as nested run
        with mlflow.start_run(nested=True, run_name=f"cv-fold-{i}"):
            for k, v in metrics.items():
                mlflow.log_metric(f"fold_{k}", v)
            mlflow.log_params({
                "fold": i,
                "train_size": len(split["train_idx"]),
                "val_size": len(split["val_idx"]),
                "train_end": str(split["train_end"]),
                "val_start": str(split["val_start"]),
                "val_end": str(split["val_end"]),
            })

        fold_metrics.append(metrics)
        fold_mapes.append(metrics["mape"])

    # Aggregate
    mean_mape = float(np.mean(fold_mapes))
    std_mape = float(np.std(fold_mapes))
    cv_passed = std_mape <= max_mape_std

    mlflow.log_metric("cv_mean_mape", mean_mape)
    mlflow.log_metric("cv_std_mape", std_mape)
    mlflow.log_metric("cv_n_folds", len(splits))
    mlflow.set_tag("cv_passed", str(cv_passed))

    if not cv_passed:
        logger.warning(
            "CV stability check FAILED: std(MAPE)=%.4f > threshold=%.4f. "
            "Model predictions are unstable across time windows.",
            std_mape, max_mape_std,
        )
    else:
        logger.info(
            "CV stability check passed: std(MAPE)=%.4f <= threshold=%.4f",
            std_mape, max_mape_std,
        )

    return {
        "fold_metrics": fold_metrics,
        "fold_mapes": fold_mapes,
        "cv_mean_mape": mean_mape,
        "cv_std_mape": std_mape,
        "cv_max_mape_std_threshold": max_mape_std,
        "cv_passed": cv_passed,
        "n_folds": len(splits),
    }


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def _build_and_train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> WeightedEnsemble:
    """Build and train a weighted ensemble from ensemble config.

    Reads model definitions from ``configs/training/ensemble.yaml``,
    trains each sub-model independently, and combines them via
    :class:`WeightedEnsemble`.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Base training configuration dict (lightgbm.yaml).

    Returns:
        A fitted ``WeightedEnsemble`` instance.
    """
    ensemble_cfg = load_config(ENSEMBLE_CONFIG_PATH)
    model_defs = ensemble_cfg["ensemble"]["models"]

    trained_models: list[tuple[str, Any]] = []
    weights: list[float] = []

    for model_def in model_defs:
        mtype = model_def["type"]
        weight = model_def.get("weight", 1.0 / len(model_defs))

        # Resolve model config: either inline or referenced from base config
        if "config_key" in model_def:
            model_cfg = config[model_def["config_key"]]
        else:
            model_cfg = model_def.get("config", {})

        logger.info("Training ensemble member: %s (weight=%.2f)", mtype, weight)
        builder = get_model_builder(mtype)
        sub_model = builder(model_cfg, random_seed=config["random_seed"])

        # Train with appropriate strategy
        if isinstance(sub_model, lgb.LGBMRegressor):
            early_stopping_rounds = config["model"].get("early_stopping_rounds", 50)
            sub_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_names=["val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=50),
                ],
            )
        else:
            sub_model.fit(X_train, y_train)

        trained_models.append((mtype, sub_model))
        weights.append(weight)

    # Normalise weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    return build_ensemble(trained_models, weights)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    tune: bool = False,
    cv: bool = False,
    model_type: str = "lightgbm",
) -> str:
    """Orchestrate the full training pipeline.

    1. Load config and data.
    2. Split temporally.
    3. Optionally run walk-forward cross-validation.
    4. Optionally tune hyperparameters.
    5. Train final model (single or ensemble).
    6. Compute ML and business metrics, log to MLflow.
    7. Write ``run_id`` to ``metrics/run_id.txt`` for DVC.

    Args:
        tune: If ``True``, run Optuna before final training.
        cv: If ``True``, run walk-forward cross-validation.
        model_type: One of ``lightgbm``, ``xgboost``, ``ridge``,
            ``ensemble``.

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
        mlflow.set_tag("model_type", model_type)

        # -- optional cross-validation --
        cv_results: dict[str, Any] | None = None
        if cv:
            logger.info("Running walk-forward cross-validation ...")
            cv_results = run_cross_validation(
                df, feature_cols, target_col, cfg, model_type=model_type,
            )
            mlflow.set_tag("cv_enabled", "True")
        else:
            mlflow.set_tag("cv_enabled", "False")

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
        if model_type == "ensemble":
            logger.info("Training weighted ensemble ...")
            model = _build_and_train_ensemble(X_train, y_train, X_val, y_val, cfg)
        else:
            model = build_model(cfg, model_type=model_type)
            model = train_model(model, X_train, y_train, X_val, y_val, cfg)

        # -- ML metrics --
        val_metrics = compute_regression_metrics(y_val.values, model.predict(X_val))
        test_metrics = compute_regression_metrics(y_test.values, model.predict(X_test))

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        logger.info("Validation metrics: %s", val_metrics)
        logger.info("Test metrics: %s", test_metrics)

        # -- business metrics --
        val_biz = compute_imbalance_cost(y_val.values, model.predict(X_val))
        test_biz = compute_imbalance_cost(y_test.values, model.predict(X_test))

        for k, v in val_biz.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_biz.items():
            mlflow.log_metric(f"test_{k}", v)

        logger.info("Validation business metrics: %s", val_biz)
        logger.info("Test business metrics: %s", test_biz)

        # -- log model artifact with signature --
        from mlflow.models.signature import infer_signature

        if model_type == "ensemble":
            # Ensemble is not natively serialisable by MLflow;
            # log each sub-model individually and store weights as param
            for name, sub_model in model.models:
                if isinstance(sub_model, lgb.LGBMRegressor):
                    mlflow.lightgbm.log_model(
                        sub_model, artifact_path=f"model_{name}",
                    )
            mlflow.log_params({
                f"ensemble_weight_{name}": w
                for (name, _), w in zip(model.models, model.weights)
            })
        else:
            signature = infer_signature(
                X_train.head(5), model.predict(X_train.head(5)),
            )
            if isinstance(model, lgb.LGBMRegressor):
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.head(1),
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.head(1),
                )

        # -- train quantile models for uncertainty estimation (P10/P50/P90) --
        if model_type == "lightgbm":
            early_stopping_rounds = cfg["model"]["early_stopping_rounds"]
            model_params = {
                k: v for k, v in cfg["model"].items()
                if k not in ("objective", "early_stopping_rounds")
            }
            quantiles: dict[str, float] = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
            for name, alpha in quantiles.items():
                logger.info("Training quantile model %s (alpha=%.1f) ...", name, alpha)
                q_model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    **model_params,
                    random_state=cfg["random_seed"],
                    verbose=-1,
                )
                q_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                )
                mlflow.lightgbm.log_model(q_model, artifact_path=f"model_{name}")
                logger.info("Quantile model %s logged to MLflow", name)

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
        _write_metrics_json(
            run_id, X_train, val_metrics, test_metrics,
            val_biz=val_biz, test_biz=test_biz, cv_results=cv_results,
        )
        (METRICS_DIR / "run_id.txt").write_text(run_id)

        logger.info("Training complete. MLflow run_id=%s", run_id)
        return run_id


def _write_metrics_json(
    run_id: str,
    X_train: pd.DataFrame,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    val_biz: dict[str, float] | None = None,
    test_biz: dict[str, float] | None = None,
    cv_results: dict[str, Any] | None = None,
) -> None:
    """Persist metrics JSON for the DVC pipeline."""
    payload: dict[str, Any] = {
        "run_id": run_id,
        "training_rows": len(X_train),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    if val_biz:
        payload.update({f"val_{k}": v for k, v in val_biz.items()})
    if test_biz:
        payload.update({f"test_{k}": v for k, v in test_biz.items()})
    if cv_results:
        payload["cv_mean_mape"] = cv_results["cv_mean_mape"]
        payload["cv_std_mape"] = cv_results["cv_std_mape"]
        payload["cv_passed"] = cv_results["cv_passed"]
        payload["cv_n_folds"] = cv_results["n_folds"]
        payload["cv_fold_mapes"] = cv_results["fold_mapes"]
    with open(METRICS_DIR / "train_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m src.ml.training.train``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train energy demand forecasting model",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning before final training",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run walk-forward cross-validation before final training",
    )
    parser.add_argument(
        "--model-type",
        choices=["lightgbm", "xgboost", "ridge", "ensemble"],
        default="lightgbm",
        help="Model architecture to train (default: lightgbm)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_id = run_training_pipeline(
        tune=args.tune, cv=args.cv, model_type=args.model_type,
    )
    print(f"Training finished. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
