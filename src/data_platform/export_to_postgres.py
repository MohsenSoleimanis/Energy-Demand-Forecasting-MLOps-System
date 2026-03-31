"""Export gold tables from DuckDB to PostgreSQL for production serving."""
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

DELTA_URIS = {
    "feature_base": "s3://lakehouse/gold/feature_base/",
    "training_set": "s3://lakehouse/gold/training_set/",
}


def get_postgres_engine():
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_WAREHOUSE_DB", "energy_warehouse")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")


def export_gold_tables():
    """Export feature_base and training_set from DuckDB to PostgreSQL."""
    db_path = Path("src/data_platform/dbt_project/energy_demand.duckdb")
    con = duckdb.connect(str(db_path), read_only=True)
    engine = get_postgres_engine()

    tables = ["feature_base", "training_set", "monitoring_set"]
    for table in tables:
        try:
            df = con.execute(f"SELECT * FROM main_gold.{table}").fetchdf()
            df.to_sql(table, engine, schema="gold", if_exists="replace", index=False)
            logger.info("Exported %s to PostgreSQL: %d rows", table, len(df))

            # Also write to Delta Lake on MinIO for versioned gold tables
            if table in DELTA_URIS:
                try:
                    from src.data_platform.delta_writer import write_delta_table
                    write_delta_table(df, DELTA_URIS[table], mode="overwrite")
                    logger.info("Exported %s to Delta Lake: %s", table, DELTA_URIS[table])
                except Exception as delta_err:
                    logger.warning("Failed to write %s to Delta Lake: %s", table, delta_err)

        except Exception as e:
            logger.warning("Failed to export %s: %s", table, e)

    con.close()
    engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.shared.config import load_env_file
    load_env_file()
    export_gold_tables()
