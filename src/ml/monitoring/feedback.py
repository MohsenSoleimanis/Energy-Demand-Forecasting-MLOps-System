"""Build monitoring set by joining predictions with actual load data.

This is the ground truth feedback loop -- it measures how well the
model is actually performing in production by joining logged
predictions with realized actuals.

Uses :func:`src.shared.s3.read_parquet_files` and
:func:`src.shared.s3.list_parquet_files` for S3 operations.

Usage::

    python -m src.ml.monitoring.feedback
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.shared.config import load_config, load_env_file
from src.shared.s3 import create_s3_client, list_parquet_files, read_parquet_files

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
_CONFIGS_DIR: Path = _PROJECT_ROOT / "configs"


def _load_prediction_logger_config() -> dict[str, Any]:
    """Read prediction logger bucket/prefix from the serving config.

    Returns:
        Dict with ``bucket`` and ``prefix``.
    """
    cfg = load_config(_CONFIGS_DIR / "serving" / "api.yaml")
    pl_cfg: dict[str, Any] = cfg.get("prediction_logger", {})
    return {
        "bucket": pl_cfg.get("bucket", "lakehouse"),
        "prefix": pl_cfg.get("prefix", "predictions"),
    }


# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------


def load_predictions() -> pd.DataFrame:
    """Load prediction logs from MinIO/S3.

    Reads all parquet files under the predictions prefix configured in
    ``configs/serving/api.yaml``.

    Returns:
        DataFrame of logged predictions, or an empty DataFrame if
        none are found.
    """
    pl_cfg: dict[str, Any] = _load_prediction_logger_config()
    bucket: str = pl_cfg["bucket"]
    prefix: str = pl_cfg["prefix"]

    client = create_s3_client()

    keys: list[str] = list_parquet_files(
        bucket=bucket, prefix=prefix, client=client
    )
    if not keys:
        logger.warning(
            "No prediction logs found in s3://%s/%s", bucket, prefix
        )
        return pd.DataFrame()

    result: pd.DataFrame = read_parquet_files(
        bucket=bucket, prefix=prefix, client=client
    )
    logger.info(
        "Loaded %d predictions from %d files", len(result), len(keys)
    )
    return result


def load_actuals() -> pd.DataFrame:
    """Load actual load data from DuckDB ``feature_base``.

    Returns:
        DataFrame with ``timestamp_brussels`` and ``load_mw`` columns,
        or an empty DataFrame if the database is not available.
    """
    db_path: Path = (
        _PROJECT_ROOT / "src" / "data_platform" / "dbt_project" / "energy_demand.duckdb"
    )
    if not db_path.exists():
        logger.error(
            "DuckDB not found at %s. Run 'python run.py transform' first.",
            db_path,
        )
        return pd.DataFrame()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df: pd.DataFrame = con.execute(
            """
            SELECT timestamp_brussels, load_mw
            FROM main_gold.feature_base
            ORDER BY timestamp_brussels
            """
        ).fetchdf()
        logger.info("Loaded %d actual load records", len(df))
        return df
    finally:
        con.close()


def build_monitoring_set(
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Join predictions with actuals and compute error metrics.

    Args:
        output_path: Where to write the monitoring parquet file.
            Defaults to ``data/monitoring/monitoring_set.parquet``.

    Returns:
        Merged DataFrame with error columns, or an empty DataFrame
        if no data is available.
    """
    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "monitoring" / "monitoring_set.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: pd.DataFrame = load_predictions()
    if predictions.empty:
        logger.warning("No predictions to monitor")
        return pd.DataFrame()

    actuals: pd.DataFrame = load_actuals()
    if actuals.empty:
        logger.warning("No actual data to compare")
        return pd.DataFrame()

    # Normalize timestamps
    predictions["timestamp_brussels"] = pd.to_datetime(
        predictions["timestamp_brussels"]
    )
    actuals["timestamp_brussels"] = pd.to_datetime(
        actuals["timestamp_brussels"]
    )

    # Round to nearest hour for joining
    predictions["join_hour"] = predictions["timestamp_brussels"].dt.floor("h")
    actuals["join_hour"] = actuals["timestamp_brussels"].dt.floor("h")

    # Join
    merged: pd.DataFrame = predictions.merge(
        actuals[["join_hour", "load_mw"]].rename(
            columns={"load_mw": "actual_load_mw"}
        ),
        on="join_hour",
        how="inner",
    )

    if merged.empty:
        logger.warning("No matching timestamps between predictions and actuals")
        return pd.DataFrame()

    # Compute errors
    merged["prediction_error"] = (
        merged["predicted_load_mw"] - merged["actual_load_mw"]
    )
    merged["absolute_error"] = merged["prediction_error"].abs()
    merged["percentage_error"] = (
        merged["absolute_error"] / merged["actual_load_mw"]
    ).abs()

    # Drop helper column
    merged = merged.drop(columns=["join_hour"])

    merged.to_parquet(output_path, index=False)
    logger.info(
        "Monitoring set: %d matched predictions, MAE=%.1f MW, MAPE=%.4f",
        len(merged),
        merged["absolute_error"].mean(),
        merged["percentage_error"].mean(),
    )
    return merged


def main() -> None:
    """CLI entry point for building the monitoring set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_env_file()
    build_monitoring_set()


if __name__ == "__main__":
    main()
