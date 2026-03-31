"""Automated model rollback when production performance degrades.

Monitors recent prediction errors against the model's registered MAPE.
If realized MAPE exceeds threshold, reverts to the previous model version.
"""
import logging
import os

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def check_and_rollback(
    model_name: str = "energy-demand-forecast",
    degradation_threshold: float = 1.5,  # 1.5x the registered MAPE
    min_predictions: int = 24,  # Need at least 24 predictions to evaluate
) -> dict:
    """Check if current model is degraded and rollback if needed.

    Args:
        model_name: MLflow registered model name
        degradation_threshold: multiplier - rollback if realized_mape > registered_mape * threshold
        min_predictions: minimum predictions needed before evaluating

    Returns:
        dict with action taken and details
    """
    client = mlflow.MlflowClient()

    # Get current production model
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        return {"action": "none", "reason": "no production model found"}

    prod_run = client.get_run(prod_version.run_id)
    registered_mape = prod_run.data.metrics.get("test_mape")
    if registered_mape is None:
        return {"action": "none", "reason": "no test_mape in production run"}

    # Load recent predictions + actuals from monitoring set
    from pathlib import Path
    monitoring_path = Path(__file__).resolve().parents[3] / "data" / "monitoring" / "monitoring_set.parquet"
    if not monitoring_path.exists():
        return {"action": "none", "reason": "no monitoring data available"}

    df = pd.read_parquet(monitoring_path)
    if len(df) < min_predictions:
        return {"action": "none", "reason": f"only {len(df)} predictions, need {min_predictions}"}

    # Filter to current model version
    if "model_version" in df.columns:
        df = df[df["model_version"] == str(prod_version.version)]

    if len(df) < min_predictions:
        return {"action": "none", "reason": f"only {len(df)} predictions for current model"}

    # Compute realized MAPE
    if "percentage_error" in df.columns:
        realized_mape = df["percentage_error"].mean()
    elif "actual_load_mw" in df.columns and "predicted_load_mw" in df.columns:
        mask = df["actual_load_mw"] != 0
        realized_mape = (df.loc[mask, "predicted_load_mw"] - df.loc[mask, "actual_load_mw"]).abs().div(df.loc[mask, "actual_load_mw"]).mean()
    else:
        return {"action": "none", "reason": "missing columns for MAPE calculation"}

    mape_limit = registered_mape * degradation_threshold

    logger.info(
        "Model v%s: registered_mape=%.4f, realized_mape=%.4f, limit=%.4f",
        prod_version.version, registered_mape, realized_mape, mape_limit,
    )

    if realized_mape <= mape_limit:
        return {
            "action": "none",
            "reason": "model performing within threshold",
            "registered_mape": registered_mape,
            "realized_mape": float(realized_mape),
        }

    # ROLLBACK: Find previous version
    logger.warning(
        "Model v%s degraded: realized_mape=%.4f > limit=%.4f. Rolling back.",
        prod_version.version, realized_mape, mape_limit,
    )

    all_versions = client.search_model_versions(f"name='{model_name}'")
    all_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)

    previous = None
    for v in all_versions:
        if int(v.version) < int(prod_version.version):
            previous = v
            break

    if previous is None:
        return {
            "action": "alert",
            "reason": "degraded but no previous version to rollback to",
            "registered_mape": registered_mape,
            "realized_mape": float(realized_mape),
        }

    # Promote previous version to production
    client.set_registered_model_alias(model_name, "production", previous.version)
    # Tag the degraded version
    client.set_model_version_tag(model_name, prod_version.version, "rolled_back", "true")
    client.set_model_version_tag(model_name, prod_version.version, "rollback_reason",
                                  f"realized_mape={realized_mape:.4f}")

    logger.warning(
        "Rolled back from v%s to v%s", prod_version.version, previous.version,
    )

    return {
        "action": "rollback",
        "from_version": prod_version.version,
        "to_version": previous.version,
        "registered_mape": registered_mape,
        "realized_mape": float(realized_mape),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from src.shared.config import load_env_file
    load_env_file()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    result = check_and_rollback()
    print(f"Rollback check result: {result}")


if __name__ == "__main__":
    main()
