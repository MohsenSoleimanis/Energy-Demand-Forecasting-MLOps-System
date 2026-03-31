"""ORCH-003: Weekly Model Retraining DAG.

Runs every Sunday at 03:00 UTC. Pulls latest gold data, trains a new model,
evaluates against production, and conditionally registers if better.
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
    dag_id="retrain",
    default_args=default_args,
    description="Weekly model retraining with conditional promotion",
    schedule="0 3 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "retrain"],
) as dag:

    pull_gold = BashOperator(
        task_id="pull_gold",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.training.pull_gold import pull_gold; "
            "pull_gold()\""
        ),
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.training.train import train; "
            "train()\""
        ),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.training.evaluate import evaluate; "
            "evaluate()\""
        ),
    )

    register_if_better = BashOperator(
        task_id="register_if_better",
        bash_command=(
            "cd /app && python -c \""
            "from src.ml.training.register import register_model; "
            "register_model()\""
        ),
    )

    pull_gold >> train >> evaluate >> register_if_better
