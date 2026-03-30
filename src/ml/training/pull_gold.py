"""
Pull the gold training_set from DuckDB/MinIO to a local parquet file.

This is a helper script that makes the training data available locally
so that DVC and the training pipeline can consume it without requiring
a live database connection during every run.

Usage:
    python -m src.ml.training.pull_gold
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def pull_gold(
    output_path: str | Path | None = None,
    duckdb_path: str | None = None,
) -> Path:
    """Query gold.training_set from DuckDB and save to local parquet.

    Parameters
    ----------
    output_path : path, optional
        Where to write the parquet file.
        Defaults to ``data/gold/training_set.parquet``.
    duckdb_path : str, optional
        Path to the DuckDB database file.  Falls back to the
        ``DUCKDB_PATH`` environment variable, then to
        ``data/energy.duckdb``.

    Returns
    -------
    Path
        The path to the written parquet file.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if duckdb_path is None:
        duckdb_path = os.getenv(
            "DUCKDB_PATH",
            str(PROJECT_ROOT / "src" / "data_platform" / "dbt_project" / "energy_demand.duckdb"),
        )

    logger.info("Connecting to DuckDB at %s", duckdb_path)
    con = duckdb.connect(duckdb_path, read_only=True)

    try:
        # Install and load extensions for S3/MinIO access
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.execute(
                """
                SET s3_region='us-east-1';
                SET s3_endpoint='localhost:9000';
                SET s3_access_key_id='minioadmin';
                SET s3_secret_access_key='minioadmin';
                SET s3_use_ssl=false;
                SET s3_url_style='path';
                """
            )
        except Exception:
            pass  # Extensions may already be loaded

        # Try gold schema first, then fall back to default schema
        # Try different schema names (dbt may prefix with 'main_')
        for schema in ["main_gold", "gold", "main"]:
            try:
                query = f"SELECT * FROM {schema}.training_set ORDER BY timestamp_brussels"
                df = con.execute(query).fetchdf()
                logger.info("Found training_set in schema '%s'", schema)
                break
            except (duckdb.CatalogException, duckdb.BinderException):
                continue
        else:
            # Last resort: no schema prefix
            df = con.execute("SELECT * FROM training_set ORDER BY timestamp_brussels").fetchdf()

        logger.info(
            "Pulled %d rows, %d columns from training_set", len(df), len(df.columns)
        )

        df.to_parquet(output_path, index=False)
        logger.info("Saved training set to %s", output_path)
    finally:
        con.close()

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    path = pull_gold()
    print(f"Training set written to {path}")


if __name__ == "__main__":
    main()
