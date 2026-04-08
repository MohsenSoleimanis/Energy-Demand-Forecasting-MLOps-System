"""Model training orchestration with MLflow experiment tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
from numpy.typing import NDArray

from energy_forecast.evaluation.evaluator import ModelEvaluator
from energy_forecast.evaluation.metrics import MetricsCalculator
from energy_forecast.models.base import BaseForecaster
from energy_forecast.training.cross_validation import TimeSeriesCV
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Returned by :meth:`Trainer.train`.

    Attributes
    ----------
    run_id:
        The MLflow run ID created for this training session.
    metrics:
        Test/validation metrics dictionary.
    val_metrics:
        Validation set metrics (``None`` when no validation data supplied).
    model:
        The trained :class:`BaseForecaster` instance.
    """

    run_id: str
    metrics: dict[str, float]
    val_metrics: dict[str, float] | None
    model: BaseForecaster


@dataclass
class CVResult:
    """Returned by :meth:`Trainer.train_with_cv`.

    Attributes
    ----------
    fold_metrics:
        Per-fold metric dictionaries.
    mean_metrics:
        Element-wise mean across all folds.
    std_metrics:
        Element-wise standard deviation across all folds.
    """

    fold_metrics: list[dict[str, float]]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrate model training, evaluation, and MLflow logging.

    Parameters
    ----------
    model:
        An *untrained* :class:`BaseForecaster` instance.
    evaluator:
        A :class:`ModelEvaluator` used to score the model after training.
    config:
        Dictionary of configuration values.  Recognised keys:

        * ``experiment_name`` -- MLflow experiment name
          (default ``"energy-demand-forecasting"``).
        * ``artifact_path`` -- MLflow artifact sub-path for model storage
          (default ``"model"``).
        * ``tags`` -- dict of extra MLflow tags.
    """

    def __init__(
        self,
        model: BaseForecaster,
        evaluator: ModelEvaluator,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.config = config or {}
        self._calc = MetricsCalculator()

    # ------------------------------------------------------------------
    # Single-pass training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> TrainResult:
        """Train the model and log everything to MLflow.

        Parameters
        ----------
        X_train:
            Training feature matrix.
        y_train:
            Training target vector.
        X_val:
            Optional validation feature matrix.
        y_val:
            Optional validation target vector.

        Returns
        -------
        TrainResult
        """
        experiment_name = self.config.get("experiment_name", "energy-demand-forecasting")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("training_started", run_id=run_id, model_type=self.model.model_type)

            # Log model params
            params = self.model.get_params()
            mlflow.log_params(params)

            # Log extra tags
            tags: dict[str, str] = {
                "model_type": self.model.model_type,
            }
            tags.update(self.config.get("tags", {}))
            mlflow.set_tags(tags)

            # Fit
            start_time = time.time()
            self.model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            train_duration = time.time() - start_time
            mlflow.log_metric("train_duration_seconds", train_duration)
            logger.info("training_completed", duration=round(train_duration, 2))

            # Evaluate on training set
            y_train_pred = self.model.predict(X_train)
            train_metrics = self._calc.compute_all(y_train, y_train_pred)
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)

            # Evaluate on validation set
            val_metrics: dict[str, float] | None = None
            if X_val is not None and y_val is not None:
                y_val = np.asarray(y_val, dtype=np.float64)
                eval_result = self.evaluator.evaluate(
                    self.model, X_val, y_val, X_train=X_train, y_train=y_train
                )
                val_metrics = eval_result.metrics
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"val_{key}", value)

                # Log evaluation plots
                self.evaluator.log_to_mlflow(eval_result, run_id=run_id)
            else:
                # If no validation set, report training metrics only
                pass

            # Log dataset sizes
            mlflow.log_param("train_size", len(X_train))
            if X_val is not None:
                mlflow.log_param("val_size", len(X_val))

            # Save model artifact
            artifact_path = self.config.get("artifact_path", "model")
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_ModelWrapper(self.model),
            )
            logger.info("model_artifact_logged", artifact_path=artifact_path)

        return TrainResult(
            run_id=run_id,
            metrics=train_metrics,
            val_metrics=val_metrics,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Cross-validated training
    # ------------------------------------------------------------------

    def train_with_cv(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        cv: TimeSeriesCV,
    ) -> CVResult:
        """Train and evaluate the model using time-series cross-validation.

        A parent MLflow run is created, and each fold is a nested child run.

        Parameters
        ----------
        X:
            Full feature matrix.
        y:
            Full target vector.
        cv:
            A :class:`TimeSeriesCV` splitter.

        Returns
        -------
        CVResult
        """
        experiment_name = self.config.get("experiment_name", "energy-demand-forecasting")
        mlflow.set_experiment(experiment_name)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        fold_metrics: list[dict[str, float]] = []

        with mlflow.start_run(run_name="cross_validation") as parent_run:
            mlflow.log_params(self.model.get_params())
            mlflow.set_tag("model_type", self.model.model_type)
            mlflow.set_tag("cv_strategy", cv.strategy)
            mlflow.log_param("cv_n_splits", cv.n_splits)
            mlflow.log_param("cv_gap", cv.gap)

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                with mlflow.start_run(
                    run_name=f"fold_{fold_idx}", nested=True
                ):
                    X_fold_train, y_fold_train = X[train_idx], y[train_idx]
                    X_fold_test, y_fold_test = X[test_idx], y[test_idx]

                    # Re-initialise model params to avoid leakage between folds
                    self.model.set_params(**self.model.get_params())

                    # Fit and evaluate
                    self.model.fit(X_fold_train, y_fold_train)
                    y_fold_pred = self.model.predict(X_fold_test)
                    metrics = self._calc.compute_all(y_fold_test, y_fold_pred)

                    # Log fold metrics
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)
                    mlflow.log_metric("fold_index", fold_idx)

                    fold_metrics.append(metrics)
                    logger.info(
                        "cv_fold_complete",
                        fold=fold_idx,
                        train_size=len(train_idx),
                        test_size=len(test_idx),
                        rmse=round(metrics["rmse"], 4),
                    )

            # Aggregate across folds
            all_keys = fold_metrics[0].keys() if fold_metrics else []
            mean_metrics = {
                key: float(np.mean([m[key] for m in fold_metrics])) for key in all_keys
            }
            std_metrics = {
                key: float(np.std([m[key] for m in fold_metrics])) for key in all_keys
            }

            # Log aggregated metrics to parent run
            for key, value in mean_metrics.items():
                mlflow.log_metric(f"cv_mean_{key}", value)
            for key, value in std_metrics.items():
                mlflow.log_metric(f"cv_std_{key}", value)

            logger.info(
                "cross_validation_complete",
                n_folds=len(fold_metrics),
                mean_rmse=round(mean_metrics.get("rmse", 0.0), 4),
            )

        return CVResult(
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
        )


# ---------------------------------------------------------------------------
# MLflow pyfunc wrapper
# ---------------------------------------------------------------------------


class _ModelWrapper(mlflow.pyfunc.PythonModel):
    """Thin wrapper so that :class:`BaseForecaster` can be stored as an
    MLflow pyfunc model artifact."""

    def __init__(self, model: BaseForecaster) -> None:
        self.model = model

    def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:  # noqa: ARG002
        """Generate predictions through the wrapped forecaster."""
        return self.model.predict(np.asarray(model_input))
