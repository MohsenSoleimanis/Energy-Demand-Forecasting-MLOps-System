"""XGBoost-based energy demand forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy_forecast.models.base import BaseForecaster
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)

# Default hyper-parameters that work well for hourly energy demand data
_DEFAULTS: dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
}


class XGBoostForecaster(BaseForecaster):
    """Forecaster backed by :class:`xgboost.XGBRegressor`.

    Supports early stopping when a validation set is provided to
    :meth:`fit`.

    Parameters
    ----------
    early_stopping_rounds:
        Number of rounds without improvement on the validation metric
        before training is stopped.  Only used when validation data is
        supplied to :meth:`fit`.
    **kwargs:
        Hyper-parameters forwarded to :class:`xgboost.XGBRegressor`.
        Any parameter not supplied falls back to sensible defaults
        tuned for hourly energy data.
    """

    def __init__(
        self,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> None:
        # Merge caller overrides on top of defaults
        merged = {**_DEFAULTS, **kwargs}
        super().__init__(early_stopping_rounds=early_stopping_rounds, **merged)

        self._early_stopping_rounds = early_stopping_rounds
        self.model = xgb.XGBRegressor(**merged)
        self._feature_names: Optional[list[str]] = None

        logger.info(
            "xgboost_forecaster_init",
            n_estimators=merged.get("n_estimators"),
            max_depth=merged.get("max_depth"),
            learning_rate=merged.get("learning_rate"),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Train the XGBoost model with optional early stopping.

        Parameters
        ----------
        X_train, y_train:
            Training data.
        X_val, y_val:
            Validation data used for early stopping and reporting.

        Returns
        -------
        dict[str, float]
            Training and (optionally) validation metrics including RMSE,
            MAE, and the best iteration number.
        """
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        self.model.set_params(
            early_stopping_rounds=(
                self._early_stopping_rounds if X_val is not None else None
            ),
        )
        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True

        # Compute metrics
        train_preds = self.model.predict(X_train)
        metrics: dict[str, float] = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "train_mae": float(mean_absolute_error(y_train, train_preds)),
        }

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            metrics["val_mae"] = float(mean_absolute_error(y_val, val_preds))

        best_iter = getattr(self.model, "best_iteration", None)
        if best_iter is not None:
            metrics["best_iteration"] = float(best_iter)

        logger.info("xgboost_forecaster_fit", **metrics)
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the fitted XGBoost model.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predictions of shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model using the native XGBoost JSON format.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``models/xgb_model.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("xgboost_model_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load a previously saved XGBoost model.

        Parameters
        ----------
        path:
            Path to the saved model file.
        """
        path = Path(path)
        self.model.load_model(str(path))
        self.is_fitted = True
        logger.info("xgboost_model_loaded", path=str(path))

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self,
        importance_type: str = "weight",
    ) -> dict[str, float]:
        """Return feature importances from the fitted booster.

        Parameters
        ----------
        importance_type:
            One of ``"weight"``, ``"gain"``, or ``"cover"``.

        Returns
        -------
        dict[str, float]
            Mapping of feature name (or index) to importance score.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        booster = self.model.get_booster()
        return booster.get_score(importance_type=importance_type)
