"""Backfill failed ENTSO-E ingestion chunks from manifest files.

Reads manifest files from s3://lakehouse/bronze/_manifests/, retries
the failed date ranges, and removes resolved manifests.
"""
import json
import logging
from datetime import datetime

import pandas as pd

from src.shared.config import load_env_file
from src.shared.s3 import create_s3_client

logger = logging.getLogger(__name__)


def load_failed_manifests(bucket: str = "lakehouse") -> list[dict]:
    """Load all unresolved failure manifest files from S3.

    Args:
        bucket: S3 bucket name.

    Returns:
        List of manifest dicts, each augmented with ``_s3_key``.
    """
    s3 = create_s3_client()
    prefix = "bronze/_manifests/"

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    manifests: list[dict] = []

    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".json"):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            manifest: dict = json.loads(body)
            manifest["_s3_key"] = obj["Key"]
            manifests.append(manifest)

    return manifests


def run_backfill(bucket: str = "lakehouse") -> int:
    """Retry all failed ingestion chunks and clean up resolved manifests.

    Args:
        bucket: S3 bucket name.

    Returns:
        Number of date-range chunks successfully resolved.
    """
    from src.data_platform.ingestion.entsoe import (
        _get_client,
        fetch_generation,
        fetch_load,
        fetch_prices,
    )
    from src.shared.s3 import upload_parquet

    manifests = load_failed_manifests(bucket)
    if not manifests:
        logger.info("No failed manifests found. Nothing to backfill.")
        return 0

    s3 = create_s3_client()
    client = _get_client()
    resolved = 0

    fetch_map = {
        "load": fetch_load,
        "price": fetch_prices,
        "generation": fetch_generation,
    }

    for manifest in manifests:
        dataset: str = manifest["dataset"]
        failed_ranges: list[list[str]] = manifest["failed_ranges"]
        fetch_fn = fetch_map.get(dataset)

        if fetch_fn is None:
            logger.warning("Unknown dataset '%s' in manifest — skipping", dataset)
            continue

        logger.info("Backfilling %d ranges for %s", len(failed_ranges), dataset)

        still_failed: list[tuple[str, str]] = []
        for start_str, end_str in failed_ranges:
            start = pd.Timestamp(start_str)
            end = pd.Timestamp(end_str)
            try:
                df: pd.DataFrame = fetch_fn(client, start, end)
                if len(df) > 0:
                    for date_str, part in df.groupby(
                        df["timestamp_utc"].dt.strftime("%Y-%m-%d")
                    ):
                        key = f"bronze/entsoe_{dataset}/date={date_str}/data.parquet"
                        upload_parquet(part, bucket, key)
                    resolved += 1
            except Exception as exc:
                logger.warning(
                    "Backfill still failing for %s %s-%s: %s",
                    dataset, start, end, exc,
                )
                still_failed.append((start_str, end_str))

        s3_key: str = manifest["_s3_key"]
        if not still_failed:
            s3.delete_object(Bucket=bucket, Key=s3_key)
            logger.info("Resolved all ranges for %s, deleted manifest", dataset)
        else:
            updated = {
                "dataset": dataset,
                "failed_ranges": still_failed,
                "last_retry": datetime.utcnow().isoformat(),
            }
            s3.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=json.dumps(updated).encode(),
            )

    return resolved


def main() -> None:
    """CLI entry point: run backfill and print summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_env_file()
    resolved = run_backfill()
    print(f"Backfill complete. {resolved} ranges resolved.")


if __name__ == "__main__":
    main()
