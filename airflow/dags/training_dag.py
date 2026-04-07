"""Weekly model retraining DAG.

Orchestrates the full ML pipeline: data generation, validation,
feature engineering, training, evaluation, and model registration.
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
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def generate_data(**context):
    """Generate or fetch the latest energy demand data."""
    import yaml

    logger.info("Generating / fetching training data")

    with open("/opt/airflow/configs/data_config.yaml") as f:
        data_config = yaml.safe_load(f)

    from energy_forecast.data.generator import DataGenerator

    generator = DataGenerator(config=data_config)
    data_path = generator.generate()

    logger.info("Training data written to %s", data_path)
    context["ti"].xcom_push(key="data_path", value=str(data_path))


def validate_data(**context):
    """Run data quality checks with Great Expectations."""
    data_path = context["ti"].xcom_pull(task_ids="generate_data", key="data_path")
    logger.info("Validating data at %s", data_path)

    from energy_forecast.data.validator import DataValidator

    validator = DataValidator()
    report = validator.validate(data_path)

    if not report["success"]:
        raise ValueError(f"Data validation failed: {report['summary']}")

    logger.info("Data validation passed")
    context["ti"].xcom_push(key="validation_report", value=report)


def engineer_features(**context):
    """Apply feature engineering pipeline."""
    data_path = context["ti"].xcom_pull(task_ids="generate_data", key="data_path")
    logger.info("Engineering features from %s", data_path)

    import yaml

    with open("/opt/airflow/configs/model_config.yaml") as f:
        model_config = yaml.safe_load(f)

    from energy_forecast.features.engine import FeatureEngine

    engine = FeatureEngine(config=model_config)
    features_path = engine.build_features(data_path)

    logger.info("Features written to %s", features_path)
    context["ti"].xcom_push(key="features_path", value=str(features_path))


def train_model(**context):
    """Train model and log to MLflow."""
    import os

    import yaml

    features_path = context["ti"].xcom_pull(task_ids="engineer_features", key="features_path")
    logger.info("Training model on features at %s", features_path)

    with open("/opt/airflow/configs/model_config.yaml") as f:
        model_config = yaml.safe_load(f)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    from energy_forecast.training.trainer import ModelTrainer

    trainer = ModelTrainer(config=model_config, tracking_uri=mlflow_uri)
    run_id = trainer.train(features_path)

    logger.info("Training complete — MLflow run_id=%s", run_id)
    context["ti"].xcom_push(key="run_id", value=run_id)


def evaluate_model(**context):
    """Evaluate the trained model against holdout data."""
    import os

    run_id = context["ti"].xcom_pull(task_ids="train_model", key="run_id")
    features_path = context["ti"].xcom_pull(task_ids="engineer_features", key="features_path")
    logger.info("Evaluating model run_id=%s", run_id)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    from energy_forecast.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(tracking_uri=mlflow_uri)
    metrics = evaluator.evaluate(run_id=run_id, data_path=features_path)

    logger.info("Evaluation metrics: %s", metrics)
    context["ti"].xcom_push(key="metrics", value=metrics)


def register_model(**context):
    """Register the model in MLflow Model Registry if it passes quality gates."""
    import os

    run_id = context["ti"].xcom_pull(task_ids="train_model", key="run_id")
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")
    logger.info("Registering model run_id=%s", run_id)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    import mlflow

    mlflow.set_tracking_uri(mlflow_uri)

    model_name = "energy-demand-forecast"
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(
        "Registered model %s version %s",
        result.name,
        result.version,
    )

    # Transition to staging
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging",
    )
    logger.info("Model version %s transitioned to Staging", result.version)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="energy_demand_training",
    default_args=default_args,
    description="Weekly retraining pipeline for the energy demand forecast model",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "energy-forecast"],
    max_active_runs=1,
) as dag:
    t_generate = PythonOperator(
        task_id="generate_data",
        python_callable=generate_data,
    )

    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    t_features = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    t_generate >> t_validate >> t_features >> t_train >> t_evaluate >> t_register
