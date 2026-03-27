"""
Ingest actual load, day-ahead prices, and generation by fuel type from ENTSO-E
for Belgium. Data is stored as Parquet in MinIO bronze layer.
"""

import argparse
import io
import logging
import os
import time
from datetime import datetime, timedelta

import boto3
import pandas as pd
from entsoe import EntsoePandasClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

COUNTRY_CODE = "BE"
TIMEZONE = "Europe/Brussels"
BUCKET = "lakehouse"

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 2.0


# ---------------------------------------------------------------------------
# S3 / MinIO helpers
# ---------------------------------------------------------------------------

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )


def upload_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3 = get_s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.info("Uploaded %d rows to s3://%s/%s", len(df), bucket, key)


# ---------------------------------------------------------------------------
# Retry wrapper with exponential backoff
# ---------------------------------------------------------------------------

def _call_with_retry(func, *args, **kwargs):
    """Call *func* with exponential backoff on any exception (rate-limit safe)."""
    backoff = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                logger.error("Failed after %d retries: %s", MAX_RETRIES, exc)
                raise
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs ...",
                attempt,
                MAX_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2


# ---------------------------------------------------------------------------
# ENTSO-E query functions
# ---------------------------------------------------------------------------

def _get_client() -> EntsoePandasClient:
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError("ENTSOE_API_KEY environment variable is not set")
    return EntsoePandasClient(api_key=api_key)


def fetch_load(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Query actual total load and return bronze-schema DataFrame."""
    logger.info("Fetching actual load for %s -> %s", start, end)
    raw = _call_with_retry(
        client.query_load, COUNTRY_CODE, start=start, end=end,
    )
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="load_mw")
    else:
        raw.columns = ["load_mw"]

    df = raw.reset_index()
    df.columns = ["timestamp_utc", "load_mw"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["area_code"] = COUNTRY_CODE
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    df["source_version"] = "entsoe-py"
    return df


def fetch_day_ahead_prices(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Query day-ahead prices and return bronze-schema DataFrame."""
    logger.info("Fetching day-ahead prices for %s -> %s", start, end)
    raw = _call_with_retry(
        client.query_day_ahead_prices, COUNTRY_CODE, start=start, end=end,
    )
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="price_eur_mwh")

    df = raw.reset_index()
    df.columns = ["timestamp_utc", "price_eur_mwh"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["area_code"] = COUNTRY_CODE
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    df["source_version"] = "entsoe-py"
    return df


def fetch_generation(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Query generation by fuel type and return bronze-schema DataFrame."""
    logger.info("Fetching generation by fuel type for %s -> %s", start, end)
    raw = _call_with_retry(
        client.query_generation, COUNTRY_CODE, start=start, end=end,
    )
    # raw is a DataFrame with MultiIndex columns (fuel_type, ...).
    # Flatten to long format.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(str(c) for c in col).strip("_") for col in raw.columns]

    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Melt fuel-type columns into long format
    fuel_cols = [c for c in df.columns if c != "timestamp_utc"]
    df = df.melt(
        id_vars=["timestamp_utc"],
        value_vars=fuel_cols,
        var_name="fuel_type",
        value_name="generation_mw",
    )
    df["area_code"] = COUNTRY_CODE
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    df["source_version"] = "entsoe-py"
    return df


# ---------------------------------------------------------------------------
# Partitioned upload
# ---------------------------------------------------------------------------

def upload_partitioned_by_date(
    df: pd.DataFrame,
    prefix: str,
    ts_col: str = "timestamp_utc",
) -> None:
    """Write one Parquet file per date partition (idempotent overwrite)."""
    df = df.copy()
    df["_date"] = pd.to_datetime(df[ts_col]).dt.strftime("%Y-%m-%d")
    for date_str, part in df.groupby("_date"):
        key = f"{prefix}date={date_str}/data.parquet"
        upload_parquet_to_s3(part.drop(columns=["_date"]), BUCKET, key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest ENTSO-E data for Belgium into MinIO bronze layer.",
    )
    default_end = datetime.utcnow().strftime("%Y-%m-%d")
    default_start = (datetime.utcnow() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help="Start date (YYYY-MM-DD). Default: 3 years ago.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.Timestamp(args.start_date, tz=TIMEZONE)
    end = pd.Timestamp(args.end_date, tz=TIMEZONE)
    logger.info("ENTSO-E ingestion: %s to %s", start, end)

    client = _get_client()

    # 1. Actual load
    try:
        load_df = fetch_load(client, start, end)
        upload_partitioned_by_date(load_df, "bronze/entsoe_load/")
        logger.info("Load ingestion complete: %d rows", len(load_df))
    except Exception:
        logger.exception("Failed to ingest load data")

    # 2. Day-ahead prices
    try:
        price_df = fetch_day_ahead_prices(client, start, end)
        upload_partitioned_by_date(price_df, "bronze/entsoe_price/")
        logger.info("Price ingestion complete: %d rows", len(price_df))
    except Exception:
        logger.exception("Failed to ingest price data")

    # 3. Generation by fuel type
    try:
        gen_df = fetch_generation(client, start, end)
        upload_partitioned_by_date(gen_df, "bronze/entsoe_generation/")
        logger.info("Generation ingestion complete: %d rows", len(gen_df))
    except Exception:
        logger.exception("Failed to ingest generation data")

    logger.info("ENTSO-E ingestion finished.")


if __name__ == "__main__":
    main()
