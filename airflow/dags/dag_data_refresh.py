"""ORCH-001: Daily Data Refresh DAG.

Runs daily at 02:00 UTC. Ingests load and weather data, validates bronze layer,
then runs dbt transformations through silver and gold layers.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _ingest_entsoe(**context):
    """Ingest load data from ENTSO-E Transparency Platform."""
    from src.data_platform.ingestion.entsoe_client import ingest_load_data

    execution_date = context["ds"]
    ingest_load_data(date=execution_date)


def _ingest_weather(**context):
    """Ingest weather data from Open-Meteo API."""
    from src.data_platform.ingestion.weather_client import ingest_weather_data

    execution_date = context["ds"]
    ingest_weather_data(date=execution_date)


def _validate_bronze(**context):
    """Run Great Expectations validation on bronze layer."""
    from src.data_platform.quality.validation import run_bronze_validation

    results = run_bronze_validation()
    if not results["success"]:
        raise ValueError(
            f"Bronze validation failed: {results.get('statistics', {})}"
        )


with DAG(
    dag_id="data_refresh",
    default_args=default_args,
    description="Daily ingestion, validation, and dbt transformation pipeline",
    schedule="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data", "ingestion", "dbt"],
) as dag:

    ingest_entsoe = PythonOperator(
        task_id="ingest_entsoe",
        python_callable=_ingest_entsoe,
    )

    ingest_weather = PythonOperator(
        task_id="ingest_weather",
        python_callable=_ingest_weather,
    )

    validate_bronze = PythonOperator(
        task_id="validate_bronze",
        python_callable=_validate_bronze,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd /app/dbt && dbt run --profiles-dir /app/dbt/profiles",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /app/dbt && dbt test --profiles-dir /app/dbt/profiles",
    )

    ingest_entsoe >> ingest_weather >> validate_bronze >> dbt_run >> dbt_test
