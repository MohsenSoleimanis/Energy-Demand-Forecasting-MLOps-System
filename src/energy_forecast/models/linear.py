"""Ridge / Lasso linear baseline forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy_forecast.models.base import BaseForecaster
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


class LinearForecaster(BaseForecaster):
    """Thin wrapper around scikit-learn Ridge or Lasso regression.

    Parameters
    ----------
    model_type:
        ``"ridge"`` (default) or ``"lasso"``.
    alpha:
        Regularisation strength.
    **kwargs:
        Additional keyword arguments forwarded to the underlying sklearn
        estimator.
    """

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_type=model_type, alpha=alpha, **kwargs)

        estimator_cls = Ridge if model_type.lower() == "ridge" else Lasso
        # Pop our own params; pass the rest to sklearn
        sklearn_params = {k: v for k, v in kwargs.items() if k not in ("model_type", "alpha")}
        self.model = estimator_cls(alpha=alpha, **sklearn_params)
        self._model_type = model_type

        logger.info("linear_forecaster_init", model_type=model_type, alpha=alpha)

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
        """Fit the linear model and return training metrics.

        Parameters
        ----------
        X_train, y_train:
            Training data.
        X_val, y_val:
            Optional validation data (metrics are computed but not used for
            early-stopping since linear models train in a single pass).

        Returns
        -------
        dict[str, float]
            Keys include ``train_rmse``, ``train_mae``, and optionally
            ``val_rmse``, ``val_mae``.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        train_preds = self.model.predict(X_train)
        metrics: dict[str, float] = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "train_mae": float(mean_absolute_error(y_train, train_preds)),
        }

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            metrics["val_mae"] = float(mean_absolute_error(y_val, val_preds))

        logger.info("linear_forecaster_fit", **metrics)
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the fitted linear model.

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
        """Save the model using joblib.

        Parameters
        ----------
        path:
            File path (e.g. ``models/linear.joblib``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("linear_model_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path:
            File path to the joblib artefact.
        """
        path = Path(path)
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info("linear_model_loaded", path=str(path))

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> np.ndarray:
        """Return the model coefficients as a feature-importance proxy.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_features,)`` with the model coefficients.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return np.asarray(self.model.coef_)
