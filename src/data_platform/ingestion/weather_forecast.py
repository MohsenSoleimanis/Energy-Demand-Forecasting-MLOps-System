"""Ingest current weather forecast snapshots from the Open-Meteo forecast API.

Fetches the short-range hourly forecast for Brussels and stores it as
date-partitioned Parquet in the MinIO bronze layer.

All configuration is loaded from ``configs/data/ingestion.yaml``.
"""

from __future__ import annotations

import logging
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

def fetch_forecast(
    forecast_url: str,
    latitude: float,
    longitude: float,
    variables: list[str],
    forecast_days: int,
    tz: str,
    timeout: int,
) -> pd.DataFrame:
    """Fetch the current short-range weather forecast from Open-Meteo.

    Args:
        forecast_url: Open-Meteo forecast API endpoint.
        latitude: Location latitude.
        longitude: Location longitude.
        variables: List of hourly variable names to request.
        forecast_days: Number of forecast days.
        tz: Timezone string for the request.
        timeout: HTTP request timeout in seconds.

    Returns:
        DataFrame with hourly forecast columns plus metadata.

    Raises:
        DataPipelineError: If the API returns no hourly data.
        requests.HTTPError: On non-2xx API response.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "forecast_days": forecast_days,
        "timezone": tz,
    }

    logger.info("Requesting Open-Meteo forecast (%d-day)", forecast_days)
    resp = requests.get(forecast_url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        raise DataPipelineError("No hourly data returned from forecast API")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    forecast_made_at = pd.Timestamp.now(tz="UTC")
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["forecast_made_at"] = forecast_made_at
    df["forecast_hour_ahead"] = (
        (df["timestamp_utc"] - forecast_made_at).dt.total_seconds() / 3600.0
    ).round(2)
    df["ingestion_ts"] = forecast_made_at
    return df


# ---------------------------------------------------------------------------
# Partitioned upload
# ---------------------------------------------------------------------------

def _upload_partitioned_by_forecast_date(
    df: pd.DataFrame,
    bucket: str,
    prefix: str,
) -> None:
    """Write Parquet partitioned by ``forecast_date=YYYY-MM-DD/`` (idempotent).

    Args:
        df: Forecast DataFrame to partition and upload.
        bucket: S3 bucket name.
        prefix: S3 key prefix.
    """
    if df.empty:
        return
    df = df.copy()
    df["_forecast_date"] = pd.to_datetime(df["timestamp_utc"]).dt.strftime("%Y-%m-%d")

    for date_str, part in df.groupby("_forecast_date"):
        key = f"{prefix}forecast_date={date_str}/data.parquet"
        upload_parquet(part.drop(columns=["_forecast_date"]), bucket, key)
        logger.info("Uploaded %d rows to s3://%s/%s", len(part), bucket, key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate weather forecast ingestion.

    Reads endpoint, coordinates, and forecast settings from
    ``configs/data/ingestion.yaml``.
    """
    cfg = _load_ingestion_config()
    weather_cfg = cfg["weather"]
    storage_cfg = cfg["storage"]

    forecast_url: str = weather_cfg["forecast_url"]
    latitude: float = weather_cfg["latitude"]
    longitude: float = weather_cfg["longitude"]
    variables: list[str] = weather_cfg["variables"]
    forecast_days: int = weather_cfg["forecast_days"]
    tz: str = weather_cfg["timezone"]
    timeout: int = weather_cfg["request_timeout_seconds"]
    bucket: str = storage_cfg["bucket"]
    prefix = f"{storage_cfg['bronze_prefix']}/weather_forecast/"

    logger.info("Weather forecast ingestion starting")
    df = fetch_forecast(
        forecast_url, latitude, longitude, variables, forecast_days, tz, timeout,
    )
    _upload_partitioned_by_forecast_date(df, bucket, prefix)
    logger.info("Forecast ingestion complete: %d rows", len(df))


if __name__ == "__main__":
    main()
