"""Ingest actual load, day-ahead prices, and generation from ENTSO-E for Belgium.

Data is fetched in monthly chunks (ENTSO-E rejects large date ranges) and
stored as date-partitioned Parquet in the MinIO bronze layer.

All configuration is loaded from ``configs/data/ingestion.yaml``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from entsoe import EntsoePandasClient

from src.shared.config import load_config, require_env
from src.shared.exceptions import ConfigError
from src.shared.s3 import upload_parquet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "ingestion.yaml"


def _load_ingestion_config() -> dict[str, Any]:
    """Load and return the ingestion configuration dictionary.

    Raises:
        ConfigError: If the config file is missing or malformed.
    """
    return load_config(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# ENTSO-E client
# ---------------------------------------------------------------------------

def _get_client() -> EntsoePandasClient:
    """Create an ENTSO-E client using the API key from the environment.

    Raises:
        ConfigError: If ``ENTSOE_API_KEY`` is not set.
    """
    try:
        api_key = require_env("ENTSOE_API_KEY")
    except Exception as exc:
        raise ConfigError(
            "ENTSOE_API_KEY environment variable is required. "
            "Set it in your .env file or environment."
        ) from exc
    return EntsoePandasClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _call_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    backoff_seconds: float = 5.0,
    **kwargs: Any,
) -> Any:
    """Call *func* with exponential back-off on any exception.

    Args:
        func: Callable to invoke.
        max_attempts: Maximum number of tries.
        backoff_seconds: Initial back-off delay (doubles each retry).

    Returns:
        Whatever *func* returns on success.

    Raises:
        Exception: Re-raises the last exception after all retries are exhausted.
    """
    backoff = backoff_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts:
                logger.error("Failed after %d retries: %s", max_attempts, exc)
                raise
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs ...",
                attempt, max_attempts, exc, backoff,
            )
            time.sleep(backoff)
            backoff *= 2


# ---------------------------------------------------------------------------
# Fetch functions — each returns a bronze-schema DataFrame
# ---------------------------------------------------------------------------

def fetch_load(
    client: EntsoePandasClient,
    area_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retry_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch actual total load from ENTSO-E.

    Args:
        client: Authenticated ENTSO-E client.
        area_code: Country / bidding-zone code (e.g. ``"BE"``).
        start: Start of the query range (timezone-aware).
        end: End of the query range (timezone-aware).
        retry_cfg: Dict with ``max_attempts`` and ``backoff_seconds``.

    Returns:
        DataFrame with columns ``timestamp_utc``, ``load_mw``, ``area_code``,
        ``ingestion_ts``, ``source_version``.
    """
    retry_cfg = retry_cfg or {}
    logger.info("Fetching actual load %s -> %s", start, end)
    raw = _call_with_retry(
        client.query_load, area_code, start=start, end=end, **retry_cfg,
    )
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="load_mw")
    else:
        raw.columns = ["load_mw"]

    df = raw.reset_index()
    df.columns = ["timestamp_utc", "load_mw"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["area_code"] = area_code
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    df["source_version"] = "entsoe-py"
    return df


def fetch_prices(
    client: EntsoePandasClient,
    area_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retry_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch day-ahead prices from ENTSO-E.

    Args:
        client: Authenticated ENTSO-E client.
        area_code: Country / bidding-zone code.
        start: Start of the query range (timezone-aware).
        end: End of the query range (timezone-aware).
        retry_cfg: Dict with ``max_attempts`` and ``backoff_seconds``.

    Returns:
        DataFrame with columns ``timestamp_utc``, ``price_eur_mwh``,
        ``area_code``, ``ingestion_ts``, ``source_version``.
    """
    retry_cfg = retry_cfg or {}
    logger.info("Fetching day-ahead prices %s -> %s", start, end)
    raw = _call_with_retry(
        client.query_day_ahead_prices, area_code, start=start, end=end,
        **retry_cfg,
    )
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="price_eur_mwh")

    df = raw.reset_index()
    df.columns = ["timestamp_utc", "price_eur_mwh"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["area_code"] = area_code
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    df["source_version"] = "entsoe-py"
    return df


def fetch_generation(
    client: EntsoePandasClient,
    area_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retry_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch generation by fuel type from ENTSO-E (melted to long format).

    Args:
        client: Authenticated ENTSO-E client.
        area_code: Country / bidding-zone code.
        start: Start of the query range (timezone-aware).
        end: End of the query range (timezone-aware).
        retry_cfg: Dict with ``max_attempts`` and ``backoff_seconds``.

    Returns:
        Long-format DataFrame with columns ``timestamp_utc``, ``fuel_type``,
        ``generation_mw``, ``area_code``, ``ingestion_ts``, ``source_version``.
    """
    retry_cfg = retry_cfg or {}
    logger.info("Fetching generation by fuel type %s -> %s", start, end)
    raw = _call_with_retry(
        client.query_generation, area_code, start=start, end=end, **retry_cfg,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join(str(c) for c in col).strip("_") for col in raw.columns
        ]

    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    fuel_cols = [c for c in df.columns if c != "timestamp_utc"]
    df = df.melt(
        id_vars=["timestamp_utc"],
        value_vars=fuel_cols,
        var_name="fuel_type",
        value_name="generation_mw",
    )
    df["area_code"] = area_code
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    df["source_version"] = "entsoe-py"
    return df


# ---------------------------------------------------------------------------
# Monthly chunking
# ---------------------------------------------------------------------------

def _generate_monthly_ranges(
    start: pd.Timestamp, end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into monthly chunks for ENTSO-E compatibility.

    ENTSO-E rejects requests spanning more than ~1 year for some endpoints.
    Monthly chunks keep requests small and reliable.
    """
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current < end:
        month_end = (
            (current + pd.offsets.MonthEnd(1)).normalize()
            + pd.Timedelta(days=1)
        )
        if month_end.tzinfo is None:
            month_end = month_end.tz_localize(current.tzinfo)
        chunk_end = min(month_end, end)
        ranges.append((current, chunk_end))
        current = chunk_end
    return ranges


# ---------------------------------------------------------------------------
# Partitioned upload
# ---------------------------------------------------------------------------

def _upload_partitioned_by_date(
    df: pd.DataFrame,
    prefix: str,
    bucket: str,
) -> None:
    """Write one Parquet file per date partition (idempotent overwrite).

    Args:
        df: DataFrame to partition and upload.
        prefix: S3 key prefix (e.g. ``"bronze/entsoe_load/"``).
        bucket: S3 bucket name.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["timestamp_utc"]).dt.strftime("%Y-%m-%d")
    for date_str, part in df.groupby("_date"):
        key = f"{prefix}date={date_str}/data.parquet"
        upload_parquet(part.drop(columns=["_date"]), bucket, key)
        logger.info("Uploaded %d rows to s3://%s/%s", len(part), bucket, key)


# ---------------------------------------------------------------------------
# Chunked ingestion
# ---------------------------------------------------------------------------

def _ingest_chunked(
    client: EntsoePandasClient,
    fetch_fn: Callable[..., pd.DataFrame],
    prefix: str,
    bucket: str,
    area_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retry_cfg: dict[str, Any],
    label: str,
) -> int:
    """Fetch data in monthly chunks, upload each, continue on partial failures.

    Args:
        client: Authenticated ENTSO-E client.
        fetch_fn: One of ``fetch_load``, ``fetch_prices``, ``fetch_generation``.
        prefix: S3 key prefix for the dataset.
        bucket: S3 bucket name.
        area_code: ENTSO-E country code.
        start: Range start.
        end: Range end.
        retry_cfg: Retry parameters forwarded to ``_call_with_retry``.
        label: Human-readable label for logging.

    Returns:
        Total number of rows ingested.
    """
    ranges = _generate_monthly_ranges(start, end)
    total_rows = 0
    failed_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for i, (chunk_start, chunk_end) in enumerate(ranges, 1):
        logger.info(
            "[%s] Chunk %d/%d: %s -> %s",
            label, i, len(ranges), chunk_start.date(), chunk_end.date(),
        )
        try:
            df = fetch_fn(
                client, area_code, chunk_start, chunk_end,
                retry_cfg=retry_cfg,
            )
            if len(df) > 0:
                _upload_partitioned_by_date(df, prefix, bucket)
                total_rows += len(df)
        except Exception as exc:
            logger.warning("[%s] Chunk %d failed: %s", label, i, exc)
            failed_ranges.append((chunk_start, chunk_end))
            time.sleep(2)

    if failed_ranges:
        logger.warning(
            "[%s] %d/%d chunks failed: %s",
            label, len(failed_ranges), len(ranges),
            [(str(s.date()), str(e.date())) for s, e in failed_ranges],
        )
    logger.info("[%s] Complete: %d rows ingested", label, total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate ENTSO-E data ingestion for all three datasets.

    Reads date range, area code, and retry settings from
    ``configs/data/ingestion.yaml``.
    """
    cfg = _load_ingestion_config()
    entsoe_cfg = cfg["entsoe"]
    storage_cfg = cfg["storage"]
    retry_cfg_raw = cfg["retry"]

    area_code: str = entsoe_cfg["area_code"]
    bucket: str = storage_cfg["bucket"]
    bronze: str = storage_cfg["bronze_prefix"]
    retry_cfg = {
        "max_attempts": retry_cfg_raw["max_attempts"],
        "backoff_seconds": retry_cfg_raw["backoff_seconds"],
    }

    tz = "Europe/Brussels"
    start = pd.Timestamp(entsoe_cfg["start_date"], tz=tz)
    end = pd.Timestamp(entsoe_cfg["end_date"], tz=tz)

    logger.info("ENTSO-E ingestion: %s to %s (area=%s)", start, end, area_code)
    client = _get_client()

    _ingest_chunked(
        client, fetch_load,
        f"{bronze}/entsoe_load/", bucket, area_code,
        start, end, retry_cfg, "load",
    )
    _ingest_chunked(
        client, fetch_prices,
        f"{bronze}/entsoe_price/", bucket, area_code,
        start, end, retry_cfg, "price",
    )
    _ingest_chunked(
        client, fetch_generation,
        f"{bronze}/entsoe_generation/", bucket, area_code,
        start, end, retry_cfg, "generation",
    )
    logger.info("ENTSO-E ingestion finished.")


if __name__ == "__main__":
    main()
