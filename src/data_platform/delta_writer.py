"""Write DataFrames to Delta Lake tables on MinIO/S3."""
import logging
import os

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger(__name__)


def get_storage_options() -> dict:
    return {
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_ENDPOINT_URL": os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"),
        "AWS_REGION": "us-east-1",
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def write_delta_table(df, table_uri: str, mode: str = "overwrite"):
    """Write DataFrame as a Delta table to S3/MinIO.

    Args:
        df: pandas DataFrame
        table_uri: s3://bucket/path/to/table
        mode: 'overwrite', 'append', or 'error'
    """
    table = pa.Table.from_pandas(df)
    storage_options = get_storage_options()

    write_deltalake(
        table_uri,
        table,
        mode=mode,
        storage_options=storage_options,
    )
    logger.info("Wrote Delta table to %s (%d rows, mode=%s)", table_uri, len(df), mode)


def read_delta_table(table_uri: str):
    """Read a Delta table from S3/MinIO."""
    storage_options = get_storage_options()
    dt = DeltaTable(table_uri, storage_options=storage_options)
    return dt.to_pandas()


def get_delta_history(table_uri: str) -> list[dict]:
    """Get version history of a Delta table (time travel)."""
    storage_options = get_storage_options()
    dt = DeltaTable(table_uri, storage_options=storage_options)
    return dt.history()
