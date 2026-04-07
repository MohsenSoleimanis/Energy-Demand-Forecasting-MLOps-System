"""FastAPI dependency injection – model manager and configuration."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "serving_config.yaml"
_LOCAL_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


class ModelManager:
    """Thread-safe singleton that owns the current prediction model.

    The manager first attempts to load a model from the MLflow Model Registry.
    If MLflow is unavailable (e.g. in local development) it falls back to a
    local pickle/joblib model stored under ``<project>/models/``.
    """

    _instance: ModelManager | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> ModelManager:  # noqa: D102
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._model: Any = None
                    inst._model_info: dict[str, Any] = {}
                    inst._loaded_at: float | None = None
                    cls._instance = inst
        return cls._instance

    # ── public API ────────────────────────────────────────────────────────

    def load_model(
        self,
        model_name: str = "energy-demand-forecaster",
        stage: str = "Production",
    ) -> None:
        """Load a model from the MLflow registry, falling back to local disk."""
        with self._lock:
            # Try MLflow first
            loaded = self._try_load_mlflow(model_name, stage)
            if not loaded:
                loaded = self._try_load_local(model_name)
            if not loaded:
                logger.warning(
                    "No model found in MLflow or locally – serving will return dummy predictions."
                )
                self._model = _DummyModel()
                self._model_info = {
                    "model_name": model_name,
                    "model_version": "dummy-0.0.0",
                    "model_stage": "None",
                    "metrics": {},
                    "features_used": [
                        "temperature",
                        "humidity",
                        "hour_sin",
                        "hour_cos",
                    ],
                }
            self._loaded_at = time.time()
            logger.info(
                "Model loaded: %s (version=%s)",
                self._model_info.get("model_name"),
                self._model_info.get("model_version"),
            )

    def get_model(self) -> Any:
        """Return the currently loaded model object (or raise if none)."""
        if self._model is None:
            raise RuntimeError("No model is loaded. Call load_model() first.")
        return self._model

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
        return dict(self._model_info)

    def reload_model(self) -> None:
        """Re-load the model using the last-known name/stage."""
        name = self._model_info.get("model_name", "energy-demand-forecaster")
        stage = self._model_info.get("model_stage", "Production")
        self.load_model(model_name=name, stage=stage)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_at(self) -> float | None:
        return self._loaded_at

    # ── private helpers ───────────────────────────────────────────────────

    def _try_load_mlflow(self, model_name: str, stage: str) -> bool:
        try:
            import mlflow  # noqa: WPS433
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            # Attempt to get the latest version for the given stage
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                logger.info("No MLflow model versions found for stage=%s.", stage)
                return False

            mv = versions[0]
            model_uri = f"models:/{model_name}/{stage}"
            self._model = mlflow.pyfunc.load_model(model_uri)
            self._model_info = {
                "model_name": model_name,
                "model_version": mv.version,
                "model_stage": stage,
                "metrics": self._fetch_run_metrics(client, mv.run_id),
                "features_used": self._fetch_features(client, mv.run_id),
            }
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("MLflow model load failed (%s), will try local fallback.", exc)
            return False

    def _try_load_local(self, model_name: str) -> bool:
        try:
            import joblib  # noqa: WPS433
        except ImportError:
            import pickle as joblib  # type: ignore[no-redef]

        for suffix in (".joblib", ".pkl", ".pickle"):
            path = _LOCAL_MODEL_DIR / f"{model_name}{suffix}"
            if path.exists():
                try:
                    self._model = joblib.load(path)
                    self._model_info = {
                        "model_name": model_name,
                        "model_version": "local-1.0.0",
                        "model_stage": "Local",
                        "metrics": {},
                        "features_used": [
                            "temperature",
                            "humidity",
                            "hour_sin",
                            "hour_cos",
                        ],
                    }
                    logger.info("Loaded local model from %s", path)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load local model %s: %s", path, exc)
        return False

    @staticmethod
    def _fetch_run_metrics(client: Any, run_id: str) -> dict[str, float]:
        try:
            run = client.get_run(run_id)
            return dict(run.data.metrics)
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _fetch_features(client: Any, run_id: str) -> list[str]:
        try:
            run = client.get_run(run_id)
            features_str = run.data.params.get("features", "")
            if features_str:
                return [f.strip() for f in features_str.split(",")]
        except Exception:  # noqa: BLE001
            pass
        return ["temperature", "humidity", "hour_sin", "hour_cos"]


class _DummyModel:
    """Placeholder model used when no real model is available."""

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        if hasattr(X, "__len__"):
            return np.full(len(X), 150.0)
        return np.array([150.0])


# ── FastAPI dependency functions ──────────────────────────────────────────────


def get_model_manager() -> ModelManager:
    """Return the global *ModelManager* singleton."""
    return ModelManager()


def get_config() -> dict[str, Any]:
    """Load and return the serving configuration dictionary.

    The config is read from ``configs/serving_config.yaml`` relative to the
    project root.  If the file is missing a sensible default is returned.
    """
    config_path = _DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as fh:
            return yaml.safe_load(fh) or {}
    return {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "model_name": "energy-demand-forecaster",
            "model_stage": "Production",
            "batch_size": 1000,
            "timeout": 30,
            "cors_origins": ["http://localhost:3000", "http://localhost:8080"],
        },
        "logging": {"level": "INFO", "format": "json"},
    }
