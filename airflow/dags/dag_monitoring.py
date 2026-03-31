"""ORCH-004: Daily Monitoring DAG.

Runs daily at 08:00 UTC. Generates drift and performance reports,
then checks against thresholds and alerts if issues are detected.
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
    dag_id="monitoring",
    default_args=default_args,
    description="Daily model monitoring: drift detection, performance tracking, alerting",
    schedule="0 8 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "monitoring", "drift"],
) as dag:

    drift_report = BashOperator(
        task_id="drift_report",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.monitoring.drift_report import generate_drift_report; "
            "generate_drift_report()\""
        ),
    )

    performance_report = BashOperator(
        task_id="performance_report",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.monitoring.performance_report import generate_performance_report; "
            "generate_performance_report()\""
        ),
    )

    check_and_alert = BashOperator(
        task_id="check_and_alert",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.monitoring.alerting import check_and_alert; "
            "check_and_alert()\""
        ),
    )

    rollback_check = BashOperator(
        task_id="rollback_check",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.serving.rollback import check_and_rollback; "
            "check_and_rollback()\""
        ),
    )

    drift_report >> performance_report >> check_and_alert >> rollback_check
