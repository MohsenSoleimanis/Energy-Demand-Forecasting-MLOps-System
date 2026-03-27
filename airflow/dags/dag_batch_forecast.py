"""ORCH-002: Daily Batch Forecast DAG.

Runs daily at 17:00 UTC. Generates next-day hourly energy demand forecasts
using the latest registered model.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _ingest_weather_forecast(**context):
    """Fetch weather forecast data for the next 48 hours."""
    from src.data_platform.ingestion.weather_client import ingest_weather_forecast

    forecast_data = ingest_weather_forecast(horizon_hours=48)
    context["ti"].xcom_push(key="forecast_weather_path", value=forecast_data)


def _prepare_features(**context):
    """Build feature vectors from the latest data for batch prediction."""
    from src.ml.features.feature_engineering import build_forecast_features

    weather_path = context["ti"].xcom_pull(
        task_ids="ingest_weather_forecast", key="forecast_weather_path"
    )
    features_path = build_forecast_features(weather_data_path=weather_path)
    context["ti"].xcom_push(key="features_path", value=features_path)


def _generate_predictions(**context):
    """Load the production model and generate batch predictions."""
    import mlflow

    import pandas as pd

    features_path = context["ti"].xcom_pull(
        task_ids="prepare_features", key="features_path"
    )
    features = pd.read_parquet(features_path)

    model = mlflow.pyfunc.load_model("models:/energy-demand-forecaster/Production")
    predictions = model.predict(features)

    output_path = "/tmp/predictions.parquet"
    features["prediction_mw"] = predictions
    features.to_parquet(output_path)
    context["ti"].xcom_push(key="predictions_path", value=output_path)


def _store_predictions(**context):
    """Write predictions to the data warehouse."""
    from src.data_platform.storage.warehouse import store_predictions

    predictions_path = context["ti"].xcom_pull(
        task_ids="generate_predictions", key="predictions_path"
    )
    store_predictions(predictions_path=predictions_path)


def _log_metrics(**context):
    """Log batch forecast metrics to MLflow."""
    import mlflow
    import pandas as pd

    predictions_path = context["ti"].xcom_pull(
        task_ids="generate_predictions", key="predictions_path"
    )
    df = pd.read_parquet(predictions_path)

    with mlflow.start_run(run_name=f"batch_forecast_{context['ds']}"):
        mlflow.log_metric("mean_prediction_mw", df["prediction_mw"].mean())
        mlflow.log_metric("std_prediction_mw", df["prediction_mw"].std())
        mlflow.log_metric("num_predictions", len(df))


with DAG(
    dag_id="batch_forecast",
    default_args=default_args,
    description="Daily batch energy demand forecast generation",
    schedule="0 17 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "forecast", "batch"],
) as dag:

    ingest_weather_forecast = PythonOperator(
        task_id="ingest_weather_forecast",
        python_callable=_ingest_weather_forecast,
    )

    prepare_features = PythonOperator(
        task_id="prepare_features",
        python_callable=_prepare_features,
    )

    generate_predictions = PythonOperator(
        task_id="generate_predictions",
        python_callable=_generate_predictions,
    )

    store_predictions = PythonOperator(
        task_id="store_predictions",
        python_callable=_store_predictions,
    )

    log_metrics = PythonOperator(
        task_id="log_metrics",
        python_callable=_log_metrics,
    )

    (
        ingest_weather_forecast
        >> prepare_features
        >> generate_predictions
        >> store_predictions
        >> log_metrics
    )
