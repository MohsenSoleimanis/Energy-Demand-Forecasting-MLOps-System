"""Abstract base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


class BaseForecaster(ABC):
    """Abstract interface that every forecasting model must implement.

    Subclasses wrap concrete ML frameworks (scikit-learn, XGBoost, PyTorch)
    behind a uniform API so that the training, evaluation, and serving
    layers can remain framework-agnostic.

    Parameters
    ----------
    **kwargs:
        Arbitrary hyper-parameters stored in :attr:`params` and forwarded
        to the concrete implementation.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.params: dict[str, Any] = kwargs
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Train the model on the provided data.

        Parameters
        ----------
        X_train:
            Training feature matrix of shape ``(n_samples, n_features)``.
        y_train:
            Training target vector of shape ``(n_samples,)``.
        X_val:
            Optional validation feature matrix.
        y_val:
            Optional validation target vector.

        Returns
        -------
        dict[str, float]
            Dictionary of training (and optionally validation) metrics.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predictions of shape ``(n_samples,)``.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to *path*.

        Implementations should create any parent directories as needed.
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore a previously saved model from *path*."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, Any]:
        """Return a copy of the model hyper-parameters."""
        return self.params.copy()

    def set_params(self, **params: Any) -> BaseForecaster:
        """Update hyper-parameters and return ``self`` for chaining."""
        self.params.update(params)
        return self

    @property
    def name(self) -> str:
        """Human-readable model name derived from the class name."""
        return self.__class__.__name__

    @property
    def model_type(self) -> str:
        """Short string identifier for the model type.

        Defaults to the lowercase class name. Subclasses may override.
        """
        return self.__class__.__name__.lower().replace("forecaster", "")
