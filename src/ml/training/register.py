"""
Register a trained model in MLflow Model Registry if it passes quality gates.

Quality gates:
    1. test_mape < 0.07
    2. No NaN metrics
    3. Training data >= 20 000 rows

If a current production model exists, the new model must beat it on test_mape.

Usage:
    python -m src.ml.training.register --run-id <MLFLOW_RUN_ID>
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "energy-demand-forecast"

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

GATE_MAX_MAPE = 0.07
GATE_MIN_TRAINING_ROWS = 20_000


def _load_eval_metrics() -> dict:
    metrics_path = PROJECT_ROOT / "metrics" / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found at {metrics_path}. "
            "Run evaluation first."
        )
    with open(metrics_path) as f:
        return json.load(f)


def _check_quality_gates(metrics: dict, run_id: str) -> list[str]:
    """Return list of failure messages (empty means all gates pass)."""
    failures: list[str] = []

    # Gate 1 - MAPE threshold
    test_mape = metrics.get("mape", float("nan"))
    if test_mape != test_mape:  # NaN check
        failures.append("test_mape is NaN")
    elif test_mape >= GATE_MAX_MAPE:
        failures.append(
            f"test_mape={test_mape:.4f} >= threshold {GATE_MAX_MAPE}"
        )

    # Gate 2 - No NaN in core metrics
    for key in ("mae", "rmse", "mape", "r2"):
        val = metrics.get(key)
        if val is None or val != val:  # None or NaN
            failures.append(f"Metric '{key}' is NaN or missing")

    # Gate 3 - Minimum training rows
    client = MlflowClient()
    run_data = client.get_run(run_id).data
    n_train = int(run_data.params.get("n_train", 0))
    if n_train < GATE_MIN_TRAINING_ROWS:
        failures.append(
            f"Training rows={n_train} < minimum {GATE_MIN_TRAINING_ROWS}"
        )

    return failures


def _get_production_mape(client: MlflowClient) -> float | None:
    """Get test_mape of the current champion model, if any."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(version.run_id)
        return run.data.metrics.get("test_mape")
    except Exception:
        pass

    try:
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if versions:
            run = client.get_run(versions[0].run_id)
            return run.data.metrics.get("test_mape")
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(run_id: str) -> str | None:
    """Register model if it passes quality gates.

    Returns the new version string, or None if registration was skipped.
    """
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    client = MlflowClient()

    metrics = _load_eval_metrics()

    # --- Quality gates ---
    failures = _check_quality_gates(metrics, run_id)
    if failures:
        for f in failures:
            logger.error("Quality gate FAILED: %s", f)
        logger.error("Model NOT registered.")
        return None

    logger.info("All quality gates passed.")

    # --- Compare with production ---
    new_mape = metrics["mape"]
    prod_mape = _get_production_mape(client)
    if prod_mape is not None and new_mape >= prod_mape:
        logger.warning(
            "New model MAPE (%.4f) is not better than production (%.4f). "
            "Registering as candidate only.",
            new_mape,
            prod_mape,
        )

    # --- Register ---
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version
    logger.info("Registered model version %s", version)

    # Set alias
    client.set_registered_model_alias(MODEL_NAME, "candidate", version)
    logger.info("Set alias 'candidate' on version %s", version)

    return version


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from src.shared.config import load_env_file
    load_env_file()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # Read run_id from DVC pipeline output
    run_id_file = Path(__file__).resolve().parents[3] / "metrics" / "run_id.txt"
    if run_id_file.exists():
        run_id = run_id_file.read_text().strip()
    else:
        metrics = _load_eval_metrics()
        run_id = metrics.get("run_id")

    if not run_id:
        logger.error("No run_id found")
        sys.exit(1)

    version = register(run_id)
    if version is None:
        print("Model registration SKIPPED (quality gates failed).")
        sys.exit(1)
    else:
        print(f"Model registered: {MODEL_NAME} version {version}")


if __name__ == "__main__":
    main()
