"""Unified model lifecycle service for production and shadow models.

Consolidates model loading, prediction, and shadow testing into a
single stateful service.  An instance is created during the FastAPI
lifespan and stored on ``app.state`` -- no global mutable state.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd

from src.shared.exceptions import ModelError

if TYPE_CHECKING:
    from mlflow import MlflowClient

logger = logging.getLogger(__name__)


class ModelService:
    """Manages production and shadow model lifecycle.

    Args:
        model_name: Registered model name in MLflow (e.g.
            ``"energy-demand-forecast"``).
        mlflow_client: An ``MlflowClient`` instance used to query
            the model registry.
    """

    def __init__(self, model_name: str, mlflow_client: MlflowClient) -> None:
        self.model_name: str = model_name
        self._client: MlflowClient = mlflow_client
        self.production_model: object | None = None
        self.production_version: object | None = None
        self.shadow_model: object | None = None
        self.shadow_version: object | None = None
        self.quantile_models: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_production(self) -> None:
        """Load the production model from the MLflow registry.

        Raises:
            ModelError: If no model with the ``production`` alias is
                found or loading fails.
        """
        try:
            model_uri = f"models:/{self.model_name}@production"
            self.production_model = mlflow.lightgbm.load_model(model_uri)
            self.production_version = self._client.get_model_version_by_alias(
                self.model_name, "production"
            )
            logger.info(
                "Loaded production model %s v%s",
                self.model_name,
                self.production_version.version,
            )
            # Load optional quantile models for confidence intervals
            self._load_quantile_models()
        except Exception as exc:
            self.production_model = None
            self.production_version = None
            raise ModelError(
                f"Failed to load production model '{self.model_name}': {exc}"
            ) from exc

    def load_shadow(self) -> None:
        """Load the candidate model for shadow testing.

        This is a best-effort operation: if no candidate alias exists
        the shadow fields are simply cleared.  It never raises.
        """
        try:
            model_uri = f"models:/{self.model_name}@candidate"
            self.shadow_model = mlflow.lightgbm.load_model(model_uri)
            self.shadow_version = self._client.get_model_version_by_alias(
                self.model_name, "candidate"
            )
            logger.info(
                "Loaded shadow model %s v%s",
                self.model_name,
                self.shadow_version.version,
            )
        except Exception as exc:
            logger.info("No shadow/candidate model available: %s", exc)
            self.shadow_model = None
            self.shadow_version = None

    def _load_quantile_models(self) -> None:
        """Load quantile regression models for P10/P50/P90 confidence intervals.

        This is best-effort: quantile models are optional.  If any
        model fails to load, it is silently skipped.
        """
        self.quantile_models = {}
        for q in ["p10", "p50", "p90"]:
            try:
                uri = f"models:/{self.model_name}@production-{q}"
                self.quantile_models[q] = mlflow.lightgbm.load_model(uri)
                logger.info("Loaded quantile model %s (%s)", self.model_name, q)
            except Exception:
                logger.debug("Quantile model '%s' not available (optional)", q)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> float:
        """Run the production model on *features* and return the prediction.

        Args:
            features: A single-row DataFrame with all required feature
                columns.

        Returns:
            Predicted load in MW.

        Raises:
            ModelError: If no production model is loaded.
        """
        if self.production_model is None:
            raise ModelError(
                "No production model loaded. Call load_production() or "
                "POST /model/reload first."
            )
        prediction = self.production_model.predict(features)
        return float(prediction[0])

    def predict_quantiles(self, features: pd.DataFrame) -> dict[str, float]:
        """Run quantile models and return P10/P50/P90 predictions.

        Args:
            features: A single-row DataFrame with all required feature
                columns.

        Returns:
            Dict mapping quantile names (``"p10"``, ``"p50"``, ``"p90"``)
            to predicted load values.  Only includes quantiles whose
            models are loaded.
        """
        results: dict[str, float] = {}
        for q, model in self.quantile_models.items():
            try:
                pred = model.predict(features)
                results[q] = float(pred[0])
            except Exception as exc:
                logger.debug("Quantile prediction failed for %s: %s", q, exc)
        return results

    def shadow_predict(self, features: pd.DataFrame) -> float | None:
        """Run the shadow model on *features*.

        This is a best-effort operation.  If no shadow model is loaded
        or prediction fails, ``None`` is returned.

        Args:
            features: A single-row DataFrame with all required feature
                columns.

        Returns:
            Predicted load in MW, or ``None``.
        """
        if self.shadow_model is None:
            return None
        try:
            prediction = self.shadow_model.predict(features)
            return float(prediction[0])
        except Exception as exc:
            logger.debug("Shadow prediction failed (non-critical): %s", exc)
            return None

    def shadow_predict_with_meta(
        self, features: pd.DataFrame
    ) -> dict[str, object] | None:
        """Run shadow prediction and return result with metadata.

        Args:
            features: A single-row DataFrame with all required feature
                columns.

        Returns:
            Dict with ``shadow_predicted_load_mw``,
            ``shadow_model_version``, and ``shadow_latency_s``; or
            ``None`` if shadow model is unavailable.
        """
        if self.shadow_model is None:
            return None
        try:
            start = time.time()
            prediction = self.shadow_model.predict(features)
            latency = time.time() - start
            return {
                "shadow_predicted_load_mw": float(prediction[0]),
                "shadow_model_version": self.shadow_version.version,
                "shadow_latency_s": latency,
            }
        except Exception as exc:
            logger.debug("Shadow prediction failed (non-critical): %s", exc)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Reload both production and shadow models from the registry.

        Raises:
            ModelError: If the production model cannot be loaded.
        """
        self.load_production()
        self.load_shadow()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """``True`` if a production model is loaded and available."""
        return self.production_model is not None

    @property
    def version_info(self) -> dict[str, str | None]:
        """Current model version metadata.

        Returns:
            Dict with ``production_version`` and ``shadow_version``
            (each may be ``None``).
        """
        return {
            "production_version": (
                self.production_version.version
                if self.production_version is not None
                else None
            ),
            "shadow_version": (
                self.shadow_version.version
                if self.shadow_version is not None
                else None
            ),
        }
