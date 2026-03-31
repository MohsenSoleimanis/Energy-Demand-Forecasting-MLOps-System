"""Model registration with quality gates.

Reads quality-gate thresholds from ``configs/training/lightgbm.yaml``,
checks the candidate model against them, compares with the current
production champion, and registers in the MLflow Model Registry.

Usage::

    python -m src.ml.training.register
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.shared.config import load_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "lightgbm.yaml"
METRICS_DIR = PROJECT_ROOT / "metrics"


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

def check_quality_gates(
    metrics: dict[str, float],
    config: dict,
) -> tuple[bool, list[str]]:
    """Validate *metrics* against quality-gate thresholds in *config*.

    Gates checked:
        1. ``test_mape < quality_gates.max_mape``
        2. No ``NaN`` in core metrics (mae, rmse, mape, r2).
        3. ``training_rows >= quality_gates.min_training_rows``

    Args:
        metrics: Dictionary that must contain ``mape`` and optionally
            ``training_rows``.
        config: Full training config with a ``quality_gates`` section.

    Returns:
        ``(passed, failures)`` where *passed* is ``True`` when the list
        of *failures* is empty.
    """
    gates = config["quality_gates"]
    max_mape = gates["max_mape"]
    min_rows = gates["min_training_rows"]
    failures: list[str] = []

    # Gate 1 -- MAPE threshold
    test_mape = metrics.get("mape", float("nan"))
    if test_mape != test_mape:  # NaN check
        failures.append("test_mape is NaN")
    elif test_mape >= max_mape:
        failures.append(f"test_mape={test_mape:.4f} >= threshold {max_mape}")

    # Gate 2 -- no NaN in core metrics
    for key in ("mae", "rmse", "mape", "r2"):
        val = metrics.get(key)
        if val is None or val != val:
            failures.append(f"Metric '{key}' is NaN or missing")

    # Gate 3 -- minimum training rows
    n_train = metrics.get("training_rows", 0)
    if n_train < min_rows:
        failures.append(f"Training rows={n_train} < minimum {min_rows}")

    passed = len(failures) == 0
    return passed, failures


def compare_with_production(
    client: MlflowClient,
    model_name: str,
    new_mape: float,
) -> bool:
    """Return ``True`` if *new_mape* is strictly better than the champion.

    Args:
        client: MLflow tracking client.
        model_name: Registered model name.
        new_mape: MAPE of the candidate model.

    Returns:
        ``True`` when the candidate beats the current champion (or there
        is no champion).
    """
    prod_mape = _get_production_mape(client, model_name)
    if prod_mape is None:
        logger.info("No production model found; candidate wins by default.")
        return True
    is_better = new_mape < prod_mape
    if not is_better:
        logger.warning(
            "New MAPE (%.4f) is not better than production (%.4f).",
            new_mape, prod_mape,
        )
    return is_better


def register_model(run_id: str) -> None:
    """Register the model from *run_id* if it passes quality gates.

    Reads ``run_id`` from ``metrics/run_id.txt`` when *run_id* is
    not provided.  Reads evaluation metrics from
    ``metrics/eval_metrics.json``.

    Args:
        run_id: MLflow run ID to register.

    Raises:
        FileNotFoundError: If evaluation metrics are missing.
        SystemExit: If quality gates fail.
    """
    cfg = load_config(CONFIG_PATH)
    model_name = cfg["model_name"]

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    client = MlflowClient()

    metrics = _load_eval_metrics()
    # Inject training_rows from the MLflow run params
    run_data = client.get_run(run_id).data
    metrics["training_rows"] = int(run_data.params.get("n_train", 0))

    passed, failures = check_quality_gates(metrics, cfg)
    if not passed:
        for f in failures:
            logger.error("Quality gate FAILED: %s", f)
        logger.error("Model NOT registered.")
        return

    logger.info("All quality gates passed.")

    # -- compare with production --
    beats_prod = compare_with_production(client, model_name, metrics["mape"])

    # -- register --
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version
    logger.info("Registered model version %s", version)

    alias = "candidate" if not beats_prod else "candidate"
    client.set_registered_model_alias(model_name, alias, version)
    logger.info("Set alias '%s' on version %s", alias, version)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_eval_metrics() -> dict:
    """Load evaluation metrics written by the evaluate stage."""
    metrics_path = METRICS_DIR / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found at {metrics_path}. "
            "Run evaluation first."
        )
    with open(metrics_path) as f:
        return json.load(f)


def _get_production_mape(client: MlflowClient, model_name: str) -> float | None:
    """Get test_mape of the current champion model, if any."""
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
        run = client.get_run(version.run_id)
        return run.data.metrics.get("test_mape")
    except Exception:
        pass

    try:
        versions = client.search_model_versions(
            f"name='{model_name}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if versions:
            run = client.get_run(versions[0].run_id)
            return run.data.metrics.get("test_mape")
    except Exception:
        pass

    return None


def _read_run_id() -> str:
    """Read run_id from the DVC pipeline output file."""
    run_id_file = METRICS_DIR / "run_id.txt"
    if run_id_file.exists():
        return run_id_file.read_text().strip()
    # Fallback: try eval_metrics.json
    metrics = _load_eval_metrics()
    run_id = metrics.get("run_id")
    if not run_id:
        raise FileNotFoundError("No run_id found in metrics/run_id.txt or eval_metrics.json")
    return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m src.ml.training.register``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.shared.config import load_env_file
    load_env_file()

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )

    run_id = _read_run_id()
    register_model(run_id)


if __name__ == "__main__":
    main()
