"""
Model loader for production models from MLflow registry.

SERVE-003
"""

import logging

import mlflow

logger = logging.getLogger(__name__)


def load_production_model(model_name: str = "energy-demand-forecast"):
    """
    Load model with 'production' alias from MLflow registry.

    Args:
        model_name: Name of the registered model in MLflow.

    Returns:
        Tuple of (model, model_version) where model_version is an
        MLflow ModelVersion object. Returns (None, None) if loading fails.
    """
    try:
        model_uri = f"models:/{model_name}@production"
        model = mlflow.lightgbm.load_model(model_uri)

        client = mlflow.MlflowClient()
        model_version = client.get_model_version_by_alias(model_name, "production")

        logger.info(f"Loaded model {model_name} version {model_version.version}")
        return model, model_version
    except Exception as e:
        logger.warning(f"Failed to load production model: {e}")
        return None, None
