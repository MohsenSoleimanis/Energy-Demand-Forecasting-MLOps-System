"""Daily model monitoring DAG.

Computes drift reports, checks quality thresholds, and sends alerts
if model performance has degraded beyond acceptable limits.
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
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def compute_drift_report(**context):
    """Compute a comprehensive drift report using Evidently."""
    import os

    import yaml

    logger.info("Computing drift report")

    with open("/opt/airflow/configs/monitoring_config.yaml") as f:
        mon_config = yaml.safe_load(f)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    from energy_forecast.monitoring.detector import DriftDetector

    detector = DriftDetector(config=mon_config, tracking_uri=mlflow_uri)

    execution_date = context["logical_date"]
    report = detector.compute_full_report(reference_date=execution_date)

    logger.info(
        "Drift report generated — features_drifted=%d, prediction_drift=%s",
        report.get("features_drifted", 0),
        report.get("prediction_drift", False),
    )

    context["ti"].xcom_push(key="drift_report", value=report)


def check_thresholds(**context):
    """Check drift metrics against configured thresholds."""
    import yaml

    drift_report = context["ti"].xcom_pull(task_ids="compute_drift_report", key="drift_report")

    with open("/opt/airflow/configs/monitoring_config.yaml") as f:
        mon_config = yaml.safe_load(f)

    thresholds = mon_config.get("thresholds", {})
    max_feature_drift_pct = thresholds.get("max_feature_drift_pct", 30)
    max_prediction_drift_score = thresholds.get("max_prediction_drift_score", 0.15)
    min_model_performance = thresholds.get("min_model_performance", 0.85)

    total_features = drift_report.get("total_features", 1)
    features_drifted = drift_report.get("features_drifted", 0)
    drift_pct = (features_drifted / total_features) * 100 if total_features > 0 else 0

    prediction_drift_score = drift_report.get("prediction_drift_score", 0.0)
    model_performance = drift_report.get("model_performance", 1.0)

    alerts = []

    if drift_pct > max_feature_drift_pct:
        alerts.append(
            f"Feature drift threshold exceeded: {drift_pct:.1f}% > {max_feature_drift_pct}%"
        )

    if prediction_drift_score > max_prediction_drift_score:
        alerts.append(
            f"Prediction drift threshold exceeded: "
            f"{prediction_drift_score:.4f} > {max_prediction_drift_score}"
        )

    if model_performance < min_model_performance:
        alerts.append(
            f"Model performance below threshold: "
            f"{model_performance:.4f} < {min_model_performance}"
        )

    threshold_result = {
        "drift_pct": drift_pct,
        "prediction_drift_score": prediction_drift_score,
        "model_performance": model_performance,
        "alerts": alerts,
        "degraded": len(alerts) > 0,
    }

    logger.info("Threshold check results: %s", threshold_result)
    context["ti"].xcom_push(key="threshold_result", value=threshold_result)


def alert_if_degraded(**context):
    """Send alerts if model performance has degraded."""
    import yaml

    threshold_result = context["ti"].xcom_pull(task_ids="check_thresholds", key="threshold_result")

    if not threshold_result.get("degraded"):
        logger.info("All metrics within acceptable thresholds — no alerts needed")
        return

    alerts = threshold_result["alerts"]
    logger.warning("Model degradation detected — %d alert(s) triggered", len(alerts))

    with open("/opt/airflow/configs/monitoring_config.yaml") as f:
        mon_config = yaml.safe_load(f)

    alert_config = mon_config.get("alerting", {})

    # Publish alert metrics to Prometheus push gateway if configured
    pushgateway_url = alert_config.get("pushgateway_url")
    if pushgateway_url:
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

            registry = CollectorRegistry()
            drift_gauge = Gauge(
                "model_drift_alert",
                "Model drift alert indicator",
                registry=registry,
            )
            drift_gauge.set(1)

            performance_gauge = Gauge(
                "model_performance_score",
                "Current model performance score",
                registry=registry,
            )
            performance_gauge.set(threshold_result.get("model_performance", 0))

            push_to_gateway(pushgateway_url, job="energy_forecast_monitoring", registry=registry)
            logger.info("Alert metrics pushed to %s", pushgateway_url)
        except Exception:
            logger.exception("Failed to push alert metrics to Prometheus")

    # Log alerts via webhook if configured
    webhook_url = alert_config.get("webhook_url")
    if webhook_url:
        try:
            import httpx

            payload = {
                "text": (
                    f"Energy Demand Forecast - Model Degradation Alert\n"
                    f"Date: {context['logical_date'].isoformat()}\n"
                    f"Alerts:\n" + "\n".join(f"  - {a}" for a in alerts)
                ),
            }
            response = httpx.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Alert sent to webhook %s", webhook_url)
        except Exception:
            logger.exception("Failed to send alert to webhook")

    for alert in alerts:
        logger.warning("ALERT: %s", alert)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="energy_demand_monitoring",
    default_args=default_args,
    description="Daily monitoring pipeline — drift detection and alerting",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "monitoring", "energy-forecast"],
    max_active_runs=1,
) as dag:
    t_drift_report = PythonOperator(
        task_id="compute_drift_report",
        python_callable=compute_drift_report,
    )

    t_check = PythonOperator(
        task_id="check_thresholds",
        python_callable=check_thresholds,
    )

    t_alert = PythonOperator(
        task_id="alert_if_degraded",
        python_callable=alert_if_degraded,
    )

    t_drift_report >> t_check >> t_alert
