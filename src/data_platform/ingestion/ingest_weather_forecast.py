"""
Ingest current weather forecast snapshots from the Open-Meteo forecast API
for Brussels, Belgium. Data is stored as Parquet in MinIO bronze layer.
"""

import argparse
import io
import logging
import os

import boto3
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
LATITUDE = 50.8503
LONGITUDE = 4.3517
BUCKET = "lakehouse"
PREFIX = "bronze/weather_forecast/"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
]


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
# Open-Meteo forecast fetcher
# ---------------------------------------------------------------------------

def fetch_forecast() -> pd.DataFrame:
    """Fetch the current 3-day weather forecast from Open-Meteo."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": 3,
        "timezone": "UTC",
    }

    logger.info("Requesting Open-Meteo forecast (3-day)")
    resp = requests.get(FORECAST_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        logger.warning("No hourly data returned from forecast API")
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Metadata columns
    forecast_made_at = pd.Timestamp.now(tz="UTC")
    df["latitude"] = LATITUDE
    df["longitude"] = LONGITUDE
    df["forecast_made_at"] = forecast_made_at
    df["forecast_hour_ahead"] = (
        (df["timestamp_utc"] - forecast_made_at).dt.total_seconds() / 3600.0
    ).round(2)
    df["ingestion_ts"] = forecast_made_at
    return df


# ---------------------------------------------------------------------------
# Partitioned upload by forecast_date
# ---------------------------------------------------------------------------

def upload_partitioned_by_forecast_date(df: pd.DataFrame) -> None:
    """Write Parquet partitioned by forecast_date=YYYY-MM-DD/ (idempotent)."""
    if df.empty:
        return
    df = df.copy()
    df["_forecast_date"] = pd.to_datetime(df["timestamp_utc"]).dt.strftime("%Y-%m-%d")

    for date_str, part in df.groupby("_forecast_date"):
        key = f"{PREFIX}forecast_date={date_str}/data.parquet"
        upload_parquet_to_s3(part.drop(columns=["_forecast_date"]), BUCKET, key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest weather forecast snapshot from Open-Meteo into MinIO bronze layer.",
    )
    # No date-range args needed -- we always fetch the current 3-day forecast.
    return parser.parse_args()


def main():
    _args = parse_args()
    logger.info("Weather forecast ingestion starting")

    try:
        df = fetch_forecast()
        if not df.empty:
            upload_partitioned_by_forecast_date(df)
            logger.info("Forecast ingestion complete: %d rows", len(df))
        else:
            logger.warning("No forecast data to ingest")
    except Exception:
        logger.exception("Failed to ingest weather forecast data")

    logger.info("Weather forecast ingestion finished.")


if __name__ == "__main__":
    main()
