"""ORCH-001: Daily Data Refresh DAG.

Runs daily at 02:00 UTC. Ingests load and weather data,
then runs dbt transformations through silver and gold layers.
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
    dag_id="data_refresh",
    default_args=default_args,
    description="Daily ingestion, validation, and dbt transformation pipeline",
    schedule="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data", "ingestion", "dbt"],
) as dag:

    ingest_entsoe = BashOperator(
        task_id="ingest_entsoe",
        bash_command="cd /app && python -m src.data_platform.ingestion.ingest_entsoe",
    )

    ingest_weather = BashOperator(
        task_id="ingest_weather",
        bash_command="cd /app && python -m src.data_platform.ingestion.ingest_weather",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd /app/dbt && dbt run --profiles-dir /app/dbt/profiles",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /app/dbt && dbt test --profiles-dir /app/dbt/profiles",
    )

    ingest_entsoe >> ingest_weather >> dbt_run >> dbt_test
