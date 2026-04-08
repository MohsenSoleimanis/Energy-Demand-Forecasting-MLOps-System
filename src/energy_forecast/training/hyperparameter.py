"""Optuna-based hyperparameter search with MLflow logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

import mlflow
import numpy as np
import optuna

from energy_forecast.evaluation.metrics import MetricsCalculator
from energy_forecast.models.base import BaseForecaster
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Result of hyperparameter search.

    Attributes
    ----------
    best_params:
        The hyperparameters that achieved the best metric.
    best_metric:
        The best (lowest) objective value found.
    study:
        The completed Optuna :class:`optuna.study.Study` object.
    all_trials:
        List of dicts summarising every trial (params + metric).
    """

    best_params: dict[str, Any]
    best_metric: float
    study: optuna.Study
    all_trials: list[dict[str, Any]]


class HyperparameterSearcher:
    """Optuna-based hyperparameter optimization for forecasting models.

    Each trial is logged as a nested MLflow run under a parent run that
    captures the overall search results.

    Parameters
    ----------
    model_class:
        The concrete forecaster *class* (not an instance) to instantiate
        per trial.
    X_train:
        Training feature matrix.
    y_train:
        Training target vector.
    X_val:
        Validation feature matrix.
    y_val:
        Validation target vector.
    config:
        Extra configuration.  Recognised keys:

        * ``experiment_name`` -- MLflow experiment name
          (default ``"hyperparameter-search"``).
        * ``fixed_params`` -- dict of params always passed to the model.
        * ``max_epochs`` -- max epochs for LSTM (default 20).
        * ``sequence_length`` -- sequence length for LSTM (default 168).
    """

    def __init__(
        self,
        model_class: Type[BaseForecaster],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_class = model_class
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64)
        self.X_val = np.asarray(X_val, dtype=np.float64)
        self.y_val = np.asarray(y_val, dtype=np.float64)
        self.config = config or {}
        self._all_trials: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Search space definitions
    # ------------------------------------------------------------------

    def _get_search_space(
        self, trial: optuna.Trial, model_type: str
    ) -> dict[str, Any]:
        """Define search space based on model type.

        Parameters
        ----------
        trial:
            Active Optuna trial object.
        model_type:
            The class name of the model (e.g. ``"LinearForecaster"``).

        Returns
        -------
        dict[str, Any]
            Sampled hyperparameters.

        Raises
        ------
        ValueError
            If *model_type* is not recognised.
        """
        if model_type == "LinearForecaster":
            return {
                "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
                "model_type": trial.suggest_categorical(
                    "model_type", ["ridge", "lasso"]
                ),
            }
        elif model_type == "XGBoostForecaster":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True
                ),
            }
        elif model_type == "LSTMForecaster":
            return {
                "input_size": (
                    self.X_train.shape[1] if self.X_train.ndim > 1 else 1
                ),
                "hidden_size": trial.suggest_categorical(
                    "hidden_size", [64, 128, 256]
                ),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128]
                ),
                "epochs": self.config.get("max_epochs", 20),
                "sequence_length": self.config.get("sequence_length", 168),
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function with MLflow nested run logging.

        Parameters
        ----------
        trial:
            Active Optuna trial.

        Returns
        -------
        float
            The objective metric value to minimise.
        """
        model_type = self.model_class.__name__
        params = self._get_search_space(trial, model_type)

        # Merge any fixed params
        fixed = self.config.get("fixed_params", {})
        merged_params = {**fixed, **params}

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log all hyperparameters
            mlflow.log_params(
                {k: v for k, v in merged_params.items() if v is not None}
            )
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("trial_number", str(trial.number))

            # Create model and train
            model = self.model_class(**merged_params)
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            predictions = model.predict(self.X_val)

            # Handle predictions that may differ in length (e.g. LSTM)
            y_eval = self.y_val
            if len(predictions) < len(y_eval):
                y_eval = y_eval[-len(predictions) :]
            elif len(predictions) > len(y_eval):
                predictions = predictions[: len(y_eval)]

            metrics = MetricsCalculator.compute_all(y_eval, predictions)

            # Log metrics to MLflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            metric_name = self.config.get("metric", "rmse")
            objective_value = metrics.get(metric_name, metrics["rmse"])
            mlflow.log_metric("objective", objective_value)

            trial_summary: dict[str, Any] = {
                "number": trial.number,
                "params": merged_params,
                "value": objective_value,
                **metrics,
            }
            self._all_trials.append(trial_summary)

            logger.info(
                "trial_complete",
                trial=trial.number,
                objective=round(objective_value, 4),
            )

        return objective_value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        metric: str = "rmse",
    ) -> SearchResult:
        """Run the hyperparameter search.

        Parameters
        ----------
        n_trials:
            Maximum number of Optuna trials.
        timeout:
            Optional timeout in seconds.  ``None`` means no time limit.
        metric:
            Name of the metric to minimise.  Must be a key returned by
            :meth:`MetricsCalculator.compute_all` (e.g. ``"rmse"``,
            ``"mae"``, ``"mape"``).

        Returns
        -------
        SearchResult
        """
        self.config["metric"] = metric
        self._all_trials = []

        experiment_name = self.config.get(
            "experiment_name", "hyperparameter-search"
        )
        mlflow.set_experiment(experiment_name)

        model_type = self.model_class.__name__

        study = optuna.create_study(
            direction="minimize",
            study_name=f"{model_type}_search",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Suppress noisy Optuna logging during search
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        with mlflow.start_run(run_name=f"hp_search_{model_type}"):
            mlflow.set_tag("search_type", "optuna")
            mlflow.set_tag("model_type", model_type)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("metric", metric)
            if timeout is not None:
                mlflow.log_param("timeout", timeout)

            study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
            )

            # Log best results to the parent run
            best_params = study.best_params
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_objective", study.best_value)

            logger.info(
                "search_complete",
                best_objective=round(study.best_value, 4),
                best_params=best_params,
                total_trials=len(study.trials),
            )

        return SearchResult(
            best_params=best_params,
            best_metric=study.best_value,
            study=study,
            all_trials=self._all_trials,
        )
