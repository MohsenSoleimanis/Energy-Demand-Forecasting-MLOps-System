"""Build monitoring set by joining predictions with actual load data.

This is the ground truth feedback loop - it lets us measure how
well the model is actually performing in production.

Usage:
    python -m src.ml.monitoring.build_monitoring_set
"""
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.shared.config import get_s3_client, load_env_file

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_predictions() -> pd.DataFrame:
    """Load prediction logs from MinIO."""
    s3 = get_s3_client()
    bucket = "lakehouse"
    prefix = "predictions/"

    # List all prediction parquet files
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        logger.warning("No prediction logs found in s3://%s/%s", bucket, prefix)
        return pd.DataFrame()

    # Read all parquet files
    dfs = []
    for obj in response["Contents"]:
        if obj["Key"].endswith(".parquet"):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            df = pd.read_parquet(pd.io.common.BytesIO(body))
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d predictions from %d files", len(result), len(dfs))
    return result


def load_actuals() -> pd.DataFrame:
    """Load actual load data from DuckDB feature_base."""
    db_path = PROJECT_ROOT / "src" / "data_platform" / "dbt_project" / "energy_demand.duckdb"
    if not db_path.exists():
        logger.error("DuckDB not found at %s. Run 'python run.py transform' first.", db_path)
        return pd.DataFrame()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT timestamp_brussels, load_mw
            FROM main_gold.feature_base
            ORDER BY timestamp_brussels
        """).fetchdf()
        logger.info("Loaded %d actual load records", len(df))
        return df
    finally:
        con.close()


def build_monitoring_set(
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Join predictions with actuals and compute error metrics."""
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "monitoring" / "monitoring_set.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    if predictions.empty:
        logger.warning("No predictions to monitor")
        return pd.DataFrame()

    actuals = load_actuals()
    if actuals.empty:
        logger.warning("No actual data to compare")
        return pd.DataFrame()

    # Normalize timestamps
    predictions["timestamp_brussels"] = pd.to_datetime(predictions["timestamp_brussels"])
    actuals["timestamp_brussels"] = pd.to_datetime(actuals["timestamp_brussels"])

    # Round to nearest hour for joining
    predictions["join_hour"] = predictions["timestamp_brussels"].dt.floor("h")
    actuals["join_hour"] = actuals["timestamp_brussels"].dt.floor("h")

    # Join
    merged = predictions.merge(
        actuals[["join_hour", "load_mw"]].rename(columns={"load_mw": "actual_load_mw"}),
        on="join_hour",
        how="inner",
    )

    if merged.empty:
        logger.warning("No matching timestamps between predictions and actuals")
        return pd.DataFrame()

    # Compute errors
    merged["prediction_error"] = merged["predicted_load_mw"] - merged["actual_load_mw"]
    merged["absolute_error"] = merged["prediction_error"].abs()
    merged["percentage_error"] = (merged["absolute_error"] / merged["actual_load_mw"]).abs()

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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_env_file()
    build_monitoring_set()


if __name__ == "__main__":
    main()
