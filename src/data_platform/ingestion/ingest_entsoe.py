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

from src.shared.config import get_s3_client


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
        raise OSError("ENTSOE_API_KEY environment variable is not set")
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


def _generate_monthly_ranges(
    start: pd.Timestamp, end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into monthly chunks for ENTSO-E API compatibility.

    ENTSO-E rejects requests spanning more than ~1 year for some endpoints.
    Monthly chunks keep requests small and reliable.
    """
    ranges = []
    current = start
    while current < end:
        month_end = (current + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(days=1)
        month_end = month_end.tz_localize(current.tzinfo) if month_end.tzinfo is None else month_end
        chunk_end = min(month_end, end)
        ranges.append((current, chunk_end))
        current = chunk_end
    return ranges


def _ingest_chunked(
    client: EntsoePandasClient,
    fetch_fn,
    prefix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str,
) -> int:
    """Fetch data in monthly chunks, upload each chunk, continue on partial failures."""
    ranges = _generate_monthly_ranges(start, end)
    total_rows = 0
    failed_ranges = []

    for i, (chunk_start, chunk_end) in enumerate(ranges, 1):
        logger.info(
            "[%s] Chunk %d/%d: %s -> %s",
            label, i, len(ranges), chunk_start.date(), chunk_end.date(),
        )
        try:
            df = fetch_fn(client, chunk_start, chunk_end)
            if len(df) > 0:
                upload_partitioned_by_date(df, prefix)
                total_rows += len(df)
        except Exception as exc:
            logger.warning("[%s] Chunk %d failed: %s", label, i, exc)
            failed_ranges.append((chunk_start, chunk_end))
            time.sleep(2)  # Brief pause before next chunk

    if failed_ranges:
        logger.warning(
            "[%s] %d/%d chunks failed: %s",
            label, len(failed_ranges), len(ranges),
            [(str(s.date()), str(e.date())) for s, e in failed_ranges],
        )
    logger.info("[%s] Complete: %d rows ingested", label, total_rows)
    return total_rows


def main():
    args = parse_args()
    start = pd.Timestamp(args.start_date, tz=TIMEZONE)
    end = pd.Timestamp(args.end_date, tz=TIMEZONE)
    logger.info("ENTSO-E ingestion: %s to %s", start, end)

    client = _get_client()

    # 1. Actual load
    _ingest_chunked(client, fetch_load, "bronze/entsoe_load/", start, end, "load")

    # 2. Day-ahead prices
    _ingest_chunked(client, fetch_day_ahead_prices, "bronze/entsoe_price/", start, end, "price")

    # 3. Generation by fuel type
    _ingest_chunked(client, fetch_generation, "bronze/entsoe_generation/", start, end, "generation")

    logger.info("ENTSO-E ingestion finished.")


if __name__ == "__main__":
    main()
