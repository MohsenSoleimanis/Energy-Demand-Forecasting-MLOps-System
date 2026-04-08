"""High-level model evaluator with plotting and MLflow integration."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from energy_forecast.evaluation.metrics import MetricsCalculator
from energy_forecast.models.base import BaseForecaster
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Container returned by :meth:`ModelEvaluator.evaluate`.

    Attributes
    ----------
    metrics:
        Dictionary of test-set metrics.
    train_metrics:
        Dictionary of training-set metrics (``None`` when no training data
        was supplied).
    residuals:
        Raw residuals ``(y_true - y_pred)`` on the test set.
    predictions:
        Model predictions on the test set.
    """

    metrics: dict[str, float]
    train_metrics: dict[str, float] | None
    residuals: NDArray[np.floating]
    predictions: NDArray[np.floating]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Evaluate a trained forecasting model and optionally log results to MLflow.

    Parameters
    ----------
    metrics_calculator:
        An optional :class:`MetricsCalculator` instance.  When ``None`` a
        default one is created automatically.
    """

    def __init__(self, metrics_calculator: MetricsCalculator | None = None) -> None:
        self._calc = metrics_calculator or MetricsCalculator()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: BaseForecaster,
        X_test: NDArray[np.floating],
        y_test: NDArray[np.floating],
        X_train: NDArray[np.floating] | None = None,
        y_train: NDArray[np.floating] | None = None,
    ) -> EvaluationResult:
        """Run full evaluation on a trained model.

        Parameters
        ----------
        model:
            A fitted :class:`BaseForecaster`.
        X_test:
            Test feature matrix.
        y_test:
            Test target vector.
        X_train:
            Optional training feature matrix (used to detect overfitting).
        y_train:
            Optional training target vector.

        Returns
        -------
        EvaluationResult
        """
        y_test = np.asarray(y_test, dtype=np.float64)
        y_pred = np.asarray(model.predict(X_test), dtype=np.float64)
        residuals = y_test - y_pred

        test_metrics = self._calc.compute_all(y_test, y_pred)
        logger.info("test_metrics", **{k: round(v, 4) for k, v in test_metrics.items()})

        # Residual analysis extras
        test_metrics["residual_mean"] = float(np.mean(residuals))
        test_metrics["residual_std"] = float(np.std(residuals))

        train_metrics: dict[str, float] | None = None
        if X_train is not None and y_train is not None:
            y_train = np.asarray(y_train, dtype=np.float64)
            y_train_pred = np.asarray(model.predict(X_train), dtype=np.float64)
            train_metrics = self._calc.compute_all(y_train, y_train_pred)
            logger.info(
                "train_metrics",
                **{k: round(v, 4) for k, v in train_metrics.items()},
            )

            # Log overfitting indicator
            overfit_ratio = (
                train_metrics["rmse"] / test_metrics["rmse"]
                if test_metrics["rmse"] > 0
                else 0.0
            )
            logger.info(
                "overfitting_check",
                train_rmse=round(train_metrics["rmse"], 4),
                test_rmse=round(test_metrics["rmse"], 4),
                ratio=round(overfit_ratio, 4),
            )

        return EvaluationResult(
            metrics=test_metrics,
            train_metrics=train_metrics,
            residuals=residuals,
            predictions=y_pred,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def generate_plots(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
        save_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Create diagnostic plots and optionally save them to disk.

        Returns a dictionary mapping plot names to ``matplotlib.figure.Figure``
        objects.  If *save_dir* is provided each figure is also saved as a PNG.

        Parameters
        ----------
        y_true:
            Ground-truth target values.
        y_pred:
            Model predictions.
        save_dir:
            Optional directory to save PNG files into.

        Returns
        -------
        dict[str, Figure]
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        residuals = y_true - y_pred

        figures: dict[str, Any] = {}

        # 1. Actual vs Predicted scatter
        fig_scatter, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.4, s=10, edgecolors="none")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        fig_scatter.tight_layout()
        figures["actual_vs_predicted"] = fig_scatter

        # 2. Residual distribution histogram
        fig_resid, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")
        fig_resid.tight_layout()
        figures["residual_distribution"] = fig_resid

        # 3. Time series overlay
        fig_ts, ax = plt.subplots(figsize=(14, 5))
        indices = np.arange(len(y_true))
        ax.plot(indices, y_true, label="Actual", alpha=0.8, linewidth=0.8)
        ax.plot(indices, y_pred, label="Predicted", alpha=0.8, linewidth=0.8)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Energy Demand")
        ax.set_title("Time Series Overlay")
        ax.legend()
        fig_ts.tight_layout()
        figures["time_series_overlay"] = fig_ts

        # 4. Error by hour-of-day boxplot
        n_samples = len(y_true)
        hours = np.arange(n_samples) % 24
        abs_errors = np.abs(residuals)

        fig_box, ax = plt.subplots(figsize=(10, 5))
        data_by_hour = [abs_errors[hours == h] for h in range(24)]
        ax.boxplot(data_by_hour, labels=[str(h) for h in range(24)], showfliers=False)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Absolute Error")
        ax.set_title("Absolute Error by Hour of Day")
        fig_box.tight_layout()
        figures["error_by_hour"] = fig_box

        # Save if requested
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(save_path / f"{name}.png", dpi=150)
            logger.info("plots_saved", directory=str(save_path), count=len(figures))

        return figures

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        evaluation_result: EvaluationResult,
        run_id: str | None = None,
    ) -> None:
        """Log evaluation metrics and diagnostic plots to an MLflow run.

        If *run_id* is ``None`` the currently active run is used.  If there
        is no active run a new one is created.

        Parameters
        ----------
        evaluation_result:
            The :class:`EvaluationResult` to log.
        run_id:
            Optional MLflow run ID.  When provided, metrics are logged to
            this specific run.
        """
        import mlflow

        def _do_log(result: EvaluationResult) -> None:
            # Log test metrics
            for key, value in result.metrics.items():
                mlflow.log_metric(f"test_{key}", value)

            # Log training metrics if available
            if result.train_metrics is not None:
                for key, value in result.train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value)

            # Generate and log plots as artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                self.generate_plots(
                    y_true=result.predictions + result.residuals,  # reconstruct y_true
                    y_pred=result.predictions,
                    save_dir=tmpdir,
                )
                mlflow.log_artifacts(tmpdir, artifact_path="evaluation_plots")

            logger.info("evaluation_logged_to_mlflow")

        if run_id is not None:
            with mlflow.start_run(run_id=run_id):
                _do_log(evaluation_result)
        else:
            # Use the active run or create a new one
            active_run = mlflow.active_run()
            if active_run is not None:
                _do_log(evaluation_result)
            else:
                with mlflow.start_run():
                    _do_log(evaluation_result)
