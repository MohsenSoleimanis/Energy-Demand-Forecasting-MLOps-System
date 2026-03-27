"""ORCH-003: Weekly Model Retraining DAG.

Runs every Sunday at 03:00 UTC. Pulls latest gold data, validates it,
trains a new model, evaluates against production, and conditionally registers.
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


def _pull_latest_gold(**context):
    """Pull the latest gold-layer training dataset."""
    from src.data_platform.storage.warehouse import export_gold_training_data

    data_path = export_gold_training_data()
    context["ti"].xcom_push(key="gold_data_path", value=data_path)


def _validate(**context):
    """Validate gold data quality before training."""
    from src.data_platform.quality.validation import run_gold_validation

    data_path = context["ti"].xcom_pull(
        task_ids="pull_latest_gold", key="gold_data_path"
    )
    results = run_gold_validation(data_path=data_path)
    if not results["success"]:
        raise ValueError(
            f"Gold data validation failed: {results.get('statistics', {})}"
        )


def _train(**context):
    """Train a new model on the validated gold dataset."""
    from src.ml.training.trainer import train_model

    data_path = context["ti"].xcom_pull(
        task_ids="pull_latest_gold", key="gold_data_path"
    )
    run_id, model_uri = train_model(data_path=data_path)
    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="model_uri", value=model_uri)


def _evaluate(**context):
    """Evaluate the new model against the current production model."""
    from src.ml.evaluation.evaluator import compare_models

    run_id = context["ti"].xcom_pull(task_ids="train", key="run_id")
    model_uri = context["ti"].xcom_pull(task_ids="train", key="model_uri")
    data_path = context["ti"].xcom_pull(
        task_ids="pull_latest_gold", key="gold_data_path"
    )

    is_better, metrics = compare_models(
        candidate_uri=model_uri,
        production_model_name="energy-demand-forecaster",
        test_data_path=data_path,
    )

    context["ti"].xcom_push(key="is_better", value=is_better)
    context["ti"].xcom_push(key="eval_metrics", value=metrics)


def _register_if_better(**context):
    """Branch: register the model only if it outperforms production."""
    is_better = context["ti"].xcom_pull(task_ids="evaluate", key="is_better")
    if is_better:
        return "register_model"
    return "notify"


def _register_model(**context):
    """Register the new model and transition it to Production stage."""
    import mlflow

    model_uri = context["ti"].xcom_pull(task_ids="train", key="model_uri")
    result = mlflow.register_model(model_uri, "energy-demand-forecaster")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="energy-demand-forecaster",
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    context["ti"].xcom_push(key="registered_version", value=result.version)


def _notify(**context):
    """Send notification about retraining results."""
    from src.ml.utils.notifications import send_retrain_notification

    eval_metrics = context["ti"].xcom_pull(task_ids="evaluate", key="eval_metrics")
    is_better = context["ti"].xcom_pull(task_ids="evaluate", key="is_better")
    registered_version = context["ti"].xcom_pull(
        task_ids="register_model", key="registered_version"
    )

    send_retrain_notification(
        is_better=is_better,
        metrics=eval_metrics,
        registered_version=registered_version,
    )


with DAG(
    dag_id="retrain",
    default_args=default_args,
    description="Weekly model retraining with conditional promotion",
    schedule="0 3 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "retrain"],
) as dag:

    pull_latest_gold = PythonOperator(
        task_id="pull_latest_gold",
        python_callable=_pull_latest_gold,
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=_validate,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=_train,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate,
    )

    register_if_better = BranchPythonOperator(
        task_id="register_if_better",
        python_callable=_register_if_better,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify,
        trigger_rule="none_failed_min_one_success",
    )

    (
        pull_latest_gold
        >> validate
        >> train
        >> evaluate
        >> register_if_better
        >> [register_model, notify]
    )
    register_model >> notify
