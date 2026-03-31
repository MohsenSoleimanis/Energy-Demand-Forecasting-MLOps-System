"""
Ingest historical weather data from the Open-Meteo archive API for Brussels,
Belgium. Data is stored as Parquet in MinIO bronze layer.
"""

import argparse
import io
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 50.8503
LONGITUDE = 4.3517
BUCKET = "lakehouse"
PREFIX = "bronze/weather_actual/"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "snow_depth",
]


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
# Open-Meteo fetcher (chunked by year to respect rate limits)
# ---------------------------------------------------------------------------

def fetch_weather_year(year: int) -> pd.DataFrame:
    """Fetch one calendar year of hourly weather data from Open-Meteo archive."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Clamp end_date to yesterday if it is in the future
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    if end_date > yesterday:
        end_date = yesterday
    if start_date > yesterday:
        logger.info("Skipping year %d (entirely in the future)", year)
        return pd.DataFrame()

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    logger.info("Requesting Open-Meteo archive: %s to %s", start_date, end_date)
    resp = requests.get(ARCHIVE_ENDPOINT, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        logger.warning("No hourly data returned for year %d", year)
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["latitude"] = LATITUDE
    df["longitude"] = LONGITUDE
    df["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
    return df


# ---------------------------------------------------------------------------
# Partitioned upload by year/month
# ---------------------------------------------------------------------------

def upload_partitioned(df: pd.DataFrame) -> None:
    """Write Parquet files partitioned by year=YYYY/month=MM/ (idempotent)."""
    if df.empty:
        return
    df = df.copy()
    df["_year"] = pd.to_datetime(df["timestamp_utc"]).dt.strftime("%Y")
    df["_month"] = pd.to_datetime(df["timestamp_utc"]).dt.strftime("%m")

    for (year, month), part in df.groupby(["_year", "_month"]):
        key = f"{PREFIX}year={year}/month={month}/data.parquet"
        upload_parquet_to_s3(part.drop(columns=["_year", "_month"]), BUCKET, key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest historical weather data from Open-Meteo into MinIO bronze layer.",
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
    start_year = int(args.start_date[:4])
    end_year = int(args.end_date[:4])

    logger.info(
        "Weather ingestion: years %d to %d (dates %s to %s)",
        start_year,
        end_year,
        args.start_date,
        args.end_date,
    )

    for year in range(start_year, end_year + 1):
        try:
            df = fetch_weather_year(year)
            if not df.empty:
                # Trim to exact requested date range
                start_ts = pd.Timestamp(args.start_date, tz="UTC")
                end_ts = pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1)
                df = df[
                    (df["timestamp_utc"] >= start_ts)
                    & (df["timestamp_utc"] < end_ts)
                ]
                upload_partitioned(df)
                logger.info("Year %d: ingested %d rows", year, len(df))
        except Exception:
            logger.exception("Failed to ingest weather data for year %d", year)

    logger.info("Weather ingestion finished.")


if __name__ == "__main__":
    main()
