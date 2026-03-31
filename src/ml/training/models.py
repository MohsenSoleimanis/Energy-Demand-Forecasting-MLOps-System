"""Model registry: build different model architectures from config.

Supports LightGBM, XGBoost, Ridge regression, and weighted ensemble.
Each builder follows a consistent interface: accept a config dict and
random seed, return an unfitted sklearn-compatible estimator.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

# Registry mapping model type strings to builder functions.
_MODEL_BUILDERS: dict[str, Any] = {}


def build_lightgbm(config: dict, random_seed: int = 42) -> lgb.LGBMRegressor:
    """Build a LightGBM regressor from config.

    Args:
        config: Model hyperparameters (n_estimators, learning_rate, etc.).
        random_seed: Reproducibility seed.

    Returns:
        Un-fitted ``LGBMRegressor``.
    """
    return lgb.LGBMRegressor(
        n_estimators=config.get("n_estimators", 1000),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 8),
        num_leaves=config.get("num_leaves", 63),
        min_child_samples=config.get("min_child_samples", 20),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 0.1),
        random_state=random_seed,
        verbose=-1,
    )


def build_xgboost(config: dict, random_seed: int = 42) -> Any:
    """Build an XGBoost regressor from config.

    Args:
        config: Model hyperparameters.
        random_seed: Reproducibility seed.

    Returns:
        Un-fitted ``XGBRegressor``.

    Raises:
        ImportError: If xgboost is not installed.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "XGBoost is required for ensemble training. "
            "Install with: pip install xgboost"
        ) from None

    return xgb.XGBRegressor(
        n_estimators=config.get("n_estimators", 1000),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 6),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 0.1),
        random_state=random_seed,
        verbosity=0,
    )


def build_ridge(config: dict, random_seed: int = 42) -> Ridge:
    """Build a Ridge regression model.

    Args:
        config: Model hyperparameters (alpha).
        random_seed: Reproducibility seed.

    Returns:
        Un-fitted ``Ridge`` regressor.
    """
    return Ridge(alpha=config.get("alpha", 1.0), random_state=random_seed)


def build_ensemble(
    models: list[tuple[str, Any]],
    weights: list[float] | None = None,
) -> WeightedEnsemble:
    """Build a weighted ensemble from trained models.

    Instead of sklearn ``VotingRegressor`` (which requires refitting),
    we use a simple ``WeightedEnsemble`` that averages predictions.

    Args:
        models: List of ``(name, fitted_model)`` tuples.
        weights: Per-model weights (must sum to 1). If ``None``,
            equal weighting is used.

    Returns:
        A ``WeightedEnsemble`` instance ready for ``.predict()``.
    """
    return WeightedEnsemble(models=models, weights=weights)


class WeightedEnsemble:
    """Simple weighted average ensemble for regression.

    Attributes:
        models: List of ``(name, fitted_model)`` tuples.
        weights: Per-model prediction weights.
    """

    def __init__(
        self,
        models: list[tuple[str, Any]],
        weights: list[float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required for an ensemble.")
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError(
                f"Got {len(self.weights)} weights for {len(self.models)} models."
            )

    def predict(self, X: Any) -> np.ndarray:
        """Generate weighted average predictions.

        Args:
            X: Feature matrix accepted by all sub-models.

        Returns:
            1-D array of predictions.
        """
        predictions = np.column_stack([
            model.predict(X) * weight
            for (name, model), weight in zip(self.models, self.weights, strict=False)
        ])
        return predictions.sum(axis=1)

    @property
    def feature_name_(self) -> list[str]:
        """Return feature names from the first model that supports it.

        Provides compatibility with LightGBM plotting utilities.
        """
        for _name, model in self.models:
            if hasattr(model, "feature_name_"):
                return model.feature_name_
        return []


# ---------------------------------------------------------------------------
# Convenience lookup
# ---------------------------------------------------------------------------

_MODEL_BUILDERS = {
    "lightgbm": build_lightgbm,
    "xgboost": build_xgboost,
    "ridge": build_ridge,
}


def get_model_builder(model_type: str):
    """Return the builder function for *model_type*.

    Args:
        model_type: One of ``lightgbm``, ``xgboost``, ``ridge``.

    Returns:
        Callable ``(config, random_seed) -> estimator``.

    Raises:
        ValueError: If *model_type* is unknown.
    """
    if model_type not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_MODEL_BUILDERS.keys())}"
        )
    return _MODEL_BUILDERS[model_type]
