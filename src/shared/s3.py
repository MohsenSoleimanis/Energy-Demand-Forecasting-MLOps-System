"""S3 / MinIO operations for the Belgian Energy Demand Forecasting system.

Every public function raises ``DataPipelineError`` on failure so callers
never have to catch low-level boto / IO exceptions.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import boto3
import pandas as pd

from src.shared.config import get_env, require_env
from src.shared.exceptions import DataPipelineError

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


def create_s3_client() -> "S3Client":
    """Create a boto3 S3 client from environment variables.

    Required env vars: ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``.
    Optional: ``S3_ENDPOINT_URL`` (defaults to ``http://localhost:9000``).

    Returns:
        A configured boto3 S3 client.

    Raises:
        DataPipelineError: If client creation fails.
    """
    try:
        return boto3.client(
            "s3",
            endpoint_url=get_env("S3_ENDPOINT_URL", "http://localhost:9000"),
            aws_access_key_id=require_env("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=require_env("AWS_SECRET_ACCESS_KEY"),
        )
    except Exception as exc:
        raise DataPipelineError(f"Failed to create S3 client: {exc}") from exc


def upload_parquet(
    df: pd.DataFrame,
    bucket: str,
    key: str,
    client: "S3Client | None" = None,
) -> None:
    """Serialize *df* as Parquet and upload to S3.

    Args:
        df: DataFrame to upload.
        bucket: S3 bucket name.
        key: Object key (path) inside the bucket.
        client: Optional pre-built S3 client; created if ``None``.

    Raises:
        DataPipelineError: On serialization or upload failure.
    """
    client = client or create_s3_client()
    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    except Exception as exc:
        raise DataPipelineError(
            f"Failed to upload Parquet to s3://{bucket}/{key}: {exc}"
        ) from exc


def list_parquet_files(
    bucket: str,
    prefix: str,
    client: "S3Client | None" = None,
) -> list[str]:
    """List all ``.parquet`` object keys under *prefix*.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix to filter on.
        client: Optional pre-built S3 client.

    Returns:
        Sorted list of matching keys.

    Raises:
        DataPipelineError: On listing failure.
    """
    client = client or create_s3_client()
    try:
        keys: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
        return sorted(keys)
    except Exception as exc:
        raise DataPipelineError(
            f"Failed to list parquet files in s3://{bucket}/{prefix}: {exc}"
        ) from exc


def read_parquet_files(
    bucket: str,
    prefix: str,
    client: "S3Client | None" = None,
) -> pd.DataFrame:
    """Read and concatenate all Parquet files under *prefix*.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix to filter on.
        client: Optional pre-built S3 client.

    Returns:
        Concatenated DataFrame.

    Raises:
        DataPipelineError: If no files are found or reading fails.
    """
    client = client or create_s3_client()
    keys = list_parquet_files(bucket, prefix, client=client)
    if not keys:
        raise DataPipelineError(
            f"No parquet files found in s3://{bucket}/{prefix}"
        )
    try:
        frames: list[pd.DataFrame] = []
        for key in keys:
            response = client.get_object(Bucket=bucket, Key=key)
            frames.append(pd.read_parquet(io.BytesIO(response["Body"].read())))
        return pd.concat(frames, ignore_index=True)
    except DataPipelineError:
        raise
    except Exception as exc:
        raise DataPipelineError(
            f"Failed to read parquet files from s3://{bucket}/{prefix}: {exc}"
        ) from exc
