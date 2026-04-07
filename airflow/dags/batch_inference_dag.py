"""Daily batch inference DAG.

Loads the latest production model, prepares input data, runs predictions,
stores results, and triggers monitoring checks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

MODEL_NAME = "energy-demand-forecast"


def load_model(**context):
    """Load the latest Production model from MLflow Model Registry."""
    import os

    import mlflow

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

    # Get the latest model version in Production (fall back to Staging)
    for stage in ("Production", "Staging"):
        versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
        if versions:
            model_version = versions[0]
            break
    else:
        raise RuntimeError(f"No model found in Production or Staging for '{MODEL_NAME}'")

    model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
    logger.info("Loading model %s (version %s, stage %s)", model_uri, model_version.version, stage)

    context["ti"].xcom_push(key="model_uri", value=model_uri)
    context["ti"].xcom_push(key="model_version", value=model_version.version)


def prepare_data(**context):
    """Prepare the latest batch of input data for inference."""
    import yaml

    logger.info("Preparing batch input data")

    with open("/opt/airflow/configs/data_config.yaml") as f:
        data_config = yaml.safe_load(f)

    from energy_forecast.data.generator import DataGenerator

    generator = DataGenerator(config=data_config)

    execution_date = context["logical_date"]
    data_path = generator.generate_batch(reference_date=execution_date)

    logger.info("Batch data written to %s", data_path)
    context["ti"].xcom_push(key="batch_data_path", value=str(data_path))


def run_predictions(**context):
    """Run batch predictions using the loaded model."""
    import os

    import mlflow
    import pandas as pd

    model_uri = context["ti"].xcom_pull(task_ids="load_model", key="model_uri")
    batch_data_path = context["ti"].xcom_pull(task_ids="prepare_data", key="batch_data_path")

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    logger.info("Reading batch data from %s", batch_data_path)
    df = pd.read_parquet(batch_data_path)

    predictions = model.predict(df)
    df["prediction"] = predictions

    output_path = batch_data_path.replace(".parquet", "_predictions.parquet")
    df.to_parquet(output_path, index=False)

    logger.info("Predictions written to %s (%d rows)", output_path, len(df))
    context["ti"].xcom_push(key="predictions_path", value=output_path)


def store_results(**context):
    """Store prediction results to the configured output location."""
    predictions_path = context["ti"].xcom_pull(task_ids="run_predictions", key="predictions_path")
    model_version = context["ti"].xcom_pull(task_ids="load_model", key="model_version")
    execution_date = context["logical_date"]

    logger.info(
        "Storing results from %s (model v%s, date %s)",
        predictions_path,
        model_version,
        execution_date.isoformat(),
    )

    from energy_forecast.pipelines.storage import ResultsStorage

    storage = ResultsStorage()
    storage.store(
        predictions_path=predictions_path,
        model_version=model_version,
        execution_date=execution_date,
    )

    logger.info("Results stored successfully")


def run_monitoring(**context):
    """Run prediction monitoring — drift and quality checks."""
    predictions_path = context["ti"].xcom_pull(task_ids="run_predictions", key="predictions_path")
    logger.info("Running monitoring checks on %s", predictions_path)

    import yaml

    with open("/opt/airflow/configs/monitoring_config.yaml") as f:
        mon_config = yaml.safe_load(f)

    from energy_forecast.monitoring.detector import DriftDetector

    detector = DriftDetector(config=mon_config)
    report = detector.analyze(predictions_path)

    if report.get("drift_detected"):
        logger.warning("Data drift detected — consider retraining. Details: %s", report)
    else:
        logger.info("No significant drift detected")

    context["ti"].xcom_push(key="monitoring_report", value=report)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="energy_demand_batch_inference",
    default_args=default_args,
    description="Daily batch prediction pipeline for energy demand forecasting",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "inference", "energy-forecast"],
    max_active_runs=1,
) as dag:
    t_load_model = PythonOperator(
        task_id="load_model",
        python_callable=load_model,
    )

    t_prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    t_run_predictions = PythonOperator(
        task_id="run_predictions",
        python_callable=run_predictions,
    )

    t_store_results = PythonOperator(
        task_id="store_results",
        python_callable=store_results,
    )

    t_run_monitoring = PythonOperator(
        task_id="run_monitoring",
        python_callable=run_monitoring,
    )

    t_load_model >> t_prepare_data >> t_run_predictions >> t_store_results >> t_run_monitoring
