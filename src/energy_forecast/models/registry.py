"""MLflow model registry wrapper for model versioning and promotion."""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional
import json
import joblib

from energy_forecast.models.base import BaseForecaster


class ModelRegistry:
    """Manages model versioning and promotion via MLflow."""

    def __init__(self, tracking_uri: str = "http://localhost:5000",
                 experiment_name: str = "energy-demand-forecasting"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        except Exception:
            # Fallback to local file store if server unavailable
            mlflow.set_tracking_uri("file:///tmp/mlruns")
            mlflow.set_experiment(self.experiment_name)

    def log_run(self, model: BaseForecaster, metrics: dict[str, float],
                params: Optional[dict[str, Any]] = None, artifacts: Optional[dict[str, str]] = None,
                tags: Optional[dict[str, str]] = None) -> str:
        """Log a training run to MLflow. Returns run_id."""
        with mlflow.start_run() as run:
            # Log parameters
            all_params = model.get_params()
            if params:
                all_params.update(params)
            for key, value in all_params.items():
                mlflow.log_param(key, value)

            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(key, float(value))

            # Log tags
            if tags:
                mlflow.set_tags(tags)
            mlflow.set_tag("model_type", model.name)

            # Save and log model artifact
            model_dir = Path(f"/tmp/mlflow_models/{run.info.run_id}")
            model.save(model_dir)
            mlflow.log_artifacts(str(model_dir), "model")

            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).is_file():
                        mlflow.log_artifact(path, name)
                    elif Path(path).is_dir():
                        mlflow.log_artifacts(path, name)

            return run.info.run_id

    def register_model(self, run_id: str, model_name: str) -> str:
        """Register a model from a run. Returns model version."""
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        return result.version

    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        """Promote a model version to a stage (Staging, Production, Archived)."""
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load a model from the registry by stage."""
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)

    def get_best_run(self, metric_name: str = "rmse", n: int = 1) -> list[dict]:
        """Get the best run(s) by a metric."""
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} ASC"],
            max_results=n,
        )
        return [
            {
                "run_id": r.info.run_id,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": r.data.tags,
                "start_time": r.info.start_time,
            }
            for r in runs
        ]

    def compare_runs(self, run_ids: list[str], metrics: list[str]) -> pd.DataFrame:
        """Compare multiple runs by specified metrics."""
        client = MlflowClient()
        rows = []
        for run_id in run_ids:
            run = client.get_run(run_id)
            row = {"run_id": run_id}
            for m in metrics:
                row[m] = run.data.metrics.get(m)
            row["model_type"] = run.data.tags.get("model_type", "unknown")
            rows.append(row)
        return pd.DataFrame(rows)
