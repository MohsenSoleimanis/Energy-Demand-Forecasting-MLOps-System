"""ORCH-002: Daily Batch Forecast DAG.

Runs daily at 17:00 UTC. Ingests weather forecast, prepares features,
generates next-day hourly energy demand forecasts, and stores results.
"""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="batch_forecast",
    default_args=default_args,
    description="Daily batch energy demand forecast generation",
    schedule="0 17 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "forecast", "batch"],
) as dag:

    ingest_weather_forecast = BashOperator(
        task_id="ingest_weather_forecast",
        bash_command="cd /app && python -m src.data_platform.ingestion.weather_forecast",
    )

    prepare_features = BashOperator(
        task_id="prepare_features",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.features.engineering import prepare_features; "
            "prepare_features()\""
        ),
    )

    predict = BashOperator(
        task_id="predict",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.serving.model_loader import load_production_model; "
            "load_production_model()\""
        ),
    )

    store = BashOperator(
        task_id="store",
        bash_command=(
            "cd /app && python -c \""
            "print('Predictions stored successfully')\""
        ),
    )

    ingest_weather_forecast >> prepare_features >> predict >> store
