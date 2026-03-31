"""Shadow model deployment - run candidate model alongside production.

The shadow model receives the same inputs as production but its
predictions are only logged, never returned to the client. This
allows safe evaluation of new models with real traffic.
"""
import logging
import time

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

_shadow_model = None
_shadow_version = None


def load_shadow_model(model_name: str = "energy-demand-forecast"):
    """Load the 'candidate' model for shadow testing."""
    global _shadow_model, _shadow_version
    try:
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(model_name, "candidate")
        model_uri = f"models:/{model_name}@candidate"
        _shadow_model = mlflow.lightgbm.load_model(model_uri)
        _shadow_version = version
        logger.info("Shadow model loaded: %s v%s", model_name, version.version)
    except Exception as e:
        logger.info("No shadow/candidate model available: %s", e)
        _shadow_model = None
        _shadow_version = None


def shadow_predict(df: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    """Run shadow prediction (non-blocking, never fails the main request).

    Returns dict with shadow prediction details, or None if no shadow model.
    """
    if _shadow_model is None:
        return None

    try:
        start = time.time()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        prediction = _shadow_model.predict(df[feature_cols])
        latency = time.time() - start

        return {
            "shadow_predicted_load_mw": float(prediction[0]),
            "shadow_model_version": _shadow_version.version,
            "shadow_latency_s": latency,
        }
    except Exception as e:
        logger.debug("Shadow prediction failed (non-critical): %s", e)
        return None
