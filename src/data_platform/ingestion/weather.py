"""Ingest historical weather data from the Open-Meteo archive API for Brussels.

Data is fetched year-by-year and stored as year/month-partitioned Parquet in
the MinIO bronze layer.

All configuration is loaded from ``configs/data/ingestion.yaml``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.shared.config import load_config
from src.shared.exceptions import DataPipelineError
from src.shared.s3 import upload_parquet

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "ingestion.yaml"


def _load_ingestion_config() -> dict[str, Any]:
    """Load and return the ingestion configuration dictionary.

    Raises:
        ConfigError: If the config file is missing or malformed.
    """
    return load_config(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_weather_year(
    year: int,
    archive_url: str,
    latitude: float,
    longitude: float,
    variables: list[str],
    tz: str,
    timeout: int,
) -> pd.DataFrame:
    """Fetch one calendar year of hourly weather data from Open-Meteo archive.

    Args:
        year: Calendar year to fetch.
        archive_url: Open-Meteo archive API endpoint.
        latitude: Location latitude.
        longitude: Location longitude.
        variables: List of hourly variable names to request.
        tz: Timezone string for the request.
        timeout: HTTP request timeout in seconds.

    Returns:
        DataFrame with hourly weather columns plus metadata. Empty if the
        year is entirely in the future.

    Raises:
        requests.HTTPError: On non-2xx API response.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    yesterday = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if start_date > yesterday:
        logger.info("Skipping year %d (entirely in the future)", year)
        return pd.DataFrame()
    if end_date > yesterday:
        end_date = yesterday

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": tz,
    }

    logger.info("Requesting Open-Meteo archive: %s to %s", start_date, end_date)
    resp = requests.get(archive_url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        raise DataPipelineError(f"No hourly data returned for year {year}")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    return df


# ---------------------------------------------------------------------------
# Partitioned upload
# ---------------------------------------------------------------------------

def upload_partitioned(
    df: pd.DataFrame,
    bucket: str,
    prefix: str,
) -> None:
    """Write Parquet files partitioned by ``year=YYYY/month=MM/`` (idempotent).

    Args:
        df: Weather DataFrame to partition and upload.
        bucket: S3 bucket name.
        prefix: S3 key prefix (e.g. ``"bronze/weather_actual/"``).
    """
    if df.empty:
        return
    df = df.copy()
    df["_year"] = pd.to_datetime(df["timestamp_utc"]).dt.strftime("%Y")
    df["_month"] = pd.to_datetime(df["timestamp_utc"]).dt.strftime("%m")

    for (year, month), part in df.groupby(["_year", "_month"]):
        key = f"{prefix}year={year}/month={month}/data.parquet"
        upload_parquet(part.drop(columns=["_year", "_month"]), bucket, key)
        logger.info("Uploaded %d rows to s3://%s/%s", len(part), bucket, key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate historical weather ingestion year-by-year.

    Reads coordinates, date range, and API settings from
    ``configs/data/ingestion.yaml``.
    """
    cfg = _load_ingestion_config()
    weather_cfg = cfg["weather"]
    storage_cfg = cfg["storage"]

    archive_url: str = weather_cfg["archive_url"]
    latitude: float = weather_cfg["latitude"]
    longitude: float = weather_cfg["longitude"]
    variables: list[str] = weather_cfg["variables"]
    tz: str = weather_cfg["timezone"]
    timeout: int = weather_cfg["request_timeout_seconds"]
    bucket: str = storage_cfg["bucket"]
    prefix = f"{storage_cfg['bronze_prefix']}/weather_actual/"

    start_year = int(weather_cfg["start_date"][:4])
    end_year = int(weather_cfg["end_date"][:4])
    start_ts = pd.Timestamp(weather_cfg["start_date"], tz="UTC")
    end_ts = pd.Timestamp(weather_cfg["end_date"], tz="UTC") + pd.Timedelta(days=1)

    logger.info("Weather ingestion: years %d to %d", start_year, end_year)

    for year in range(start_year, end_year + 1):
        try:
            df = fetch_weather_year(
                year, archive_url, latitude, longitude, variables, tz, timeout,
            )
            if not df.empty:
                df = df[
                    (df["timestamp_utc"] >= start_ts)
                    & (df["timestamp_utc"] < end_ts)
                ]
                upload_partitioned(df, bucket, prefix)
                logger.info("Year %d: ingested %d rows", year, len(df))
        except Exception:
            logger.exception("Failed to ingest weather data for year %d", year)
            raise

    logger.info("Weather ingestion finished.")


if __name__ == "__main__":
    main()
