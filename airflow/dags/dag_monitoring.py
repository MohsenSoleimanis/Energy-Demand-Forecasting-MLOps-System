"""ORCH-004: Daily Monitoring DAG.

Runs daily at 08:00 UTC. Generates drift and performance reports,
checks against thresholds, and alerts if issues are detected.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _generate_drift_report(**context):
    """Generate a data/prediction drift report using Evidently."""
    from src.ml.monitoring.drift import generate_drift_report

    report_path = generate_drift_report(reference_date=context["ds"])
    context["ti"].xcom_push(key="drift_report_path", value=report_path)


def _generate_performance_report(**context):
    """Generate a model performance report comparing predictions to actuals."""
    from src.ml.monitoring.performance import generate_performance_report

    report_path = generate_performance_report(evaluation_date=context["ds"])
    context["ti"].xcom_push(key="performance_report_path", value=report_path)


def _check_thresholds(**context):
    """Evaluate drift and performance metrics against configured thresholds."""
    from src.ml.monitoring.thresholds import check_all_thresholds

    drift_report = context["ti"].xcom_pull(
        task_ids="generate_drift_report", key="drift_report_path"
    )
    perf_report = context["ti"].xcom_pull(
        task_ids="generate_performance_report", key="performance_report_path"
    )

    violations = check_all_thresholds(
        drift_report_path=drift_report,
        performance_report_path=perf_report,
    )

    context["ti"].xcom_push(key="violations", value=violations)

    if violations:
        return "alert_if_needed"
    return "no_alert"


def _alert_if_needed(**context):
    """Send alerts for any threshold violations detected."""
    from src.ml.utils.notifications import send_monitoring_alert

    violations = context["ti"].xcom_pull(
        task_ids="check_thresholds", key="violations"
    )
    send_monitoring_alert(violations=violations, date=context["ds"])


def _no_alert(**context):
    """No-op task when all metrics are within thresholds."""
    pass


with DAG(
    dag_id="monitoring",
    default_args=default_args,
    description="Daily model monitoring: drift detection, performance tracking, alerting",
    schedule="0 8 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "monitoring", "drift"],
) as dag:

    generate_drift_report = PythonOperator(
        task_id="generate_drift_report",
        python_callable=_generate_drift_report,
    )

    generate_performance_report = PythonOperator(
        task_id="generate_performance_report",
        python_callable=_generate_performance_report,
    )

    check_thresholds = BranchPythonOperator(
        task_id="check_thresholds",
        python_callable=_check_thresholds,
    )

    alert_if_needed = PythonOperator(
        task_id="alert_if_needed",
        python_callable=_alert_if_needed,
    )

    no_alert = PythonOperator(
        task_id="no_alert",
        python_callable=_no_alert,
    )

    (
        [generate_drift_report, generate_performance_report]
        >> check_thresholds
        >> [alert_if_needed, no_alert]
    )
