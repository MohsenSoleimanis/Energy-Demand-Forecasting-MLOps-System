"""
Train a LightGBM model for Belgian energy demand forecasting.

Logs parameters, metrics, and artifacts to MLflow.

Usage:
    python -m src.ml.training.train
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap
except ImportError:
    shap = None

from src.ml.features.feature_engineering import get_feature_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load training config from Hydra YAML with sensible defaults."""
    config_path = PROJECT_ROOT / "configs" / "training" / "lightgbm.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Fallback defaults
    return {
        "model": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_seed": 42,
        },
        "split": {
            "train_end": "2024-06-30",
            "val_end": "2024-09-30",
        },
        "early_stopping_rounds": 50,
        "mlflow": {
            "experiment_name": "energy-demand-forecast",
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        },
    }


def _git_sha() -> str:
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R-squared."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Avoid division by zero in MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = float("nan")
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> str:
    """Run full training pipeline. Returns the MLflow run ID."""
    cfg = _load_config()
    model_cfg = cfg["model"]
    split_cfg = cfg["split"]
    random_seed = cfg.get("random_seed", model_cfg.get("random_seed", 42))
    early_stopping = model_cfg.get("early_stopping_rounds", cfg.get("early_stopping_rounds", 50))

    # --- Load data ---
    data_path = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run `python -m src.ml.training.pull_gold` first."
        )
    df = pd.read_parquet(data_path)
    logger.info("Loaded training data: %s rows, %s columns", len(df), len(df.columns))

    # --- Temporal split ---
    if "timestamp_brussels" in df.columns:
        df = df.sort_values("timestamp_brussels").reset_index(drop=True)
        ts = pd.to_datetime(df["timestamp_brussels"])
    else:
        raise ValueError("Column 'timestamp_brussels' not found in training data.")

    train_end = pd.Timestamp(split_cfg["train_end"])
    val_end = pd.Timestamp(split_cfg["val_end"])

    train_mask = ts <= train_end
    val_mask = (ts > train_end) & (ts <= val_end)
    test_mask = ts > val_end

    # --- Features and target ---
    target_col = "target_load_24h"
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    # Log which features are used and which are missing
    all_expected = get_feature_columns()
    missing = [c for c in all_expected if c not in df.columns]
    if missing:
        logger.warning("Missing features (will be excluded): %s", missing)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

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

    # --- MLflow ---
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg.get("experiment_name", "energy-demand-forecast"))

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log params
        mlflow.log_params(
            {
                "n_estimators": model_cfg["n_estimators"],
                "learning_rate": model_cfg["learning_rate"],
                "max_depth": model_cfg["max_depth"],
                "num_leaves": model_cfg["num_leaves"],
                "subsample": model_cfg["subsample"],
                "colsample_bytree": model_cfg["colsample_bytree"],
                "min_child_samples": model_cfg["min_child_samples"],
                "reg_alpha": model_cfg["reg_alpha"],
                "reg_lambda": model_cfg["reg_lambda"],
                "random_seed": random_seed,
                "train_end": str(split_cfg["train_end"]),
                "val_end": str(split_cfg["val_end"]),
                "n_features": len(feature_cols),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "early_stopping_rounds": early_stopping,
            }
        )

        # Tags
        mlflow.set_tag("git_commit_sha", _git_sha())
        mlflow.set_tag("python_version", platform.python_version())

        # --- Train LightGBM ---
        model = lgb.LGBMRegressor(
            n_estimators=model_cfg["n_estimators"],
            learning_rate=model_cfg["learning_rate"],
            max_depth=model_cfg["max_depth"],
            num_leaves=model_cfg["num_leaves"],
            subsample=model_cfg["subsample"],
            colsample_bytree=model_cfg["colsample_bytree"],
            min_child_samples=model_cfg["min_child_samples"],
            reg_alpha=model_cfg["reg_alpha"],
            reg_lambda=model_cfg["reg_lambda"],
            random_state=random_seed,
            verbose=-1,
        )

        # Custom callback to log per-iteration metrics to MLflow
        class MlflowLogCallback:
            """Log validation loss at each boosting round to MLflow."""
            def __init__(self):
                self.order = 25  # Run after other callbacks

            def __call__(self, env):
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    mlflow.log_metric(
                        f"{data_name}_{eval_name}",
                        result,
                        step=env.iteration,
                    )

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping),
            lgb.log_evaluation(period=50),
            MlflowLogCallback(),
        ]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["val"],
            callbacks=callbacks,
        )

        # --- Final Metrics ---
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        val_metrics = compute_metrics(y_val.values, val_preds)
        test_metrics = compute_metrics(y_test.values, test_preds)

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        logger.info("Validation metrics: %s", val_metrics)
        logger.info("Test metrics: %s", test_metrics)

        # --- Log model ---
        mlflow.lightgbm.log_model(model, artifact_path="model")

        # --- Feature importance plot ---
        plots_dir = PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        lgb.plot_importance(model, ax=ax, max_num_features=20, importance_type="gain")
        plt.tight_layout()
        fi_path = plots_dir / "feature_importance.png"
        fig.savefig(fi_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(fi_path))

        # --- SHAP summary ---
        if shap is not None:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test.head(500))
                fig = plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test.head(500), show=False)
                shap_path = plots_dir / "shap_summary.png"
                plt.savefig(shap_path, dpi=150, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(shap_path))
            except Exception as exc:
                logger.warning("SHAP summary plot failed: %s", exc)
        else:
            logger.info("SHAP not installed, skipping explainability plot")

        # --- Save metrics JSON for DVC pipeline ---
        metrics_dir = PROJECT_ROOT / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        train_metrics = {
            "run_id": run_id,
            "training_rows": int(len(X_train)),
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "val_r2": val_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_mape": test_metrics["mape"],
            "test_r2": test_metrics["r2"],
        }
        with open(metrics_dir / "train_metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=2)

        # Write run_id for DVC pipeline downstream stages
        (metrics_dir / "run_id.txt").write_text(run_id)

        logger.info("Training complete. MLflow run_id=%s", run_id)
        return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_id = train()
    print(f"Training finished. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
