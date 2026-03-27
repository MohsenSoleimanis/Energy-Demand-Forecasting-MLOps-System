"""
Generate Belgian calendar data including public holidays, school vacations,
and temporal features. Data is stored as Parquet in MinIO bronze layer.
"""

import argparse
import io
import logging
import os
from datetime import date, timedelta

import boto3
import holidays
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BUCKET = "lakehouse"
PREFIX = "bronze/calendar/"


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
# School vacation periods (approximate Flemish schedule)
# ---------------------------------------------------------------------------

def _school_vacation_ranges(year: int) -> list[tuple[date, date]]:
    """Return approximate Flemish school vacation date ranges for a given year.

    Each tuple is (start_date_inclusive, end_date_inclusive).
    """
    ranges = []

    # Christmas vacation: Dec 25 of previous year through Jan 5
    ranges.append((date(year - 1, 12, 25), date(year, 1, 5)))

    # Carnival / Crocus vacation: 1 week in late February (week containing last Monday of Feb)
    # Approximate: last full week of February
    last_day_feb = date(year, 3, 1) - timedelta(days=1)
    # Find the Monday of the last full week
    carnival_friday = last_day_feb
    while carnival_friday.weekday() != 4:  # Friday
        carnival_friday -= timedelta(days=1)
    carnival_monday = carnival_friday - timedelta(days=4)
    ranges.append((carnival_monday, carnival_friday + timedelta(days=2)))  # Mon-Sun

    # Easter vacation: 2 weeks around Easter
    # Compute Easter Sunday using the anonymous Gregorian algorithm
    easter_sunday = _compute_easter(year)
    easter_start = easter_sunday - timedelta(days=6)  # Monday before Easter
    easter_end = easter_start + timedelta(days=13)  # Two full weeks (Mon-Sun)
    ranges.append((easter_start, easter_end))

    # Summer vacation: July 1 through August 31
    ranges.append((date(year, 7, 1), date(year, 8, 31)))

    # Autumn vacation: 1 week in late October (week containing Nov 1)
    nov1 = date(year, 11, 1)
    # Find the Monday of the week containing Nov 1
    autumn_monday = nov1 - timedelta(days=nov1.weekday())
    autumn_sunday = autumn_monday + timedelta(days=6)
    ranges.append((autumn_monday, autumn_sunday))

    # Christmas vacation starting this year: Dec 25 through Dec 31
    ranges.append((date(year, 12, 25), date(year, 12, 31)))

    return ranges


def _compute_easter(year: int) -> date:
    """Compute Easter Sunday for a given year (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _is_school_vacation(d: date, ranges: list[tuple[date, date]]) -> bool:
    """Check if a date falls within any school vacation range."""
    return any(start <= d <= end for start, end in ranges)


# ---------------------------------------------------------------------------
# Calendar generation
# ---------------------------------------------------------------------------

def generate_calendar(start_year: int, end_year: int) -> pd.DataFrame:
    """Generate a daily calendar DataFrame for the given year range."""
    logger.info("Generating calendar for years %d to %d", start_year, end_year)

    year_range = range(start_year, end_year + 1)
    be_holidays = holidays.Belgium(years=year_range)

    # Pre-compute school vacation ranges for all years
    all_vacation_ranges = {}
    for year in year_range:
        all_vacation_ranges[year] = _school_vacation_ranges(year)

    rows = []
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    current = start_date

    while current <= end_date:
        holiday_name = be_holidays.get(current, "")
        is_holiday = current in be_holidays
        is_weekend = current.weekday() >= 5  # Saturday=5, Sunday=6
        is_school_vac = _is_school_vacation(
            current, all_vacation_ranges[current.year]
        )

        rows.append({
            "date": current.isoformat(),
            "is_belgian_holiday": is_holiday,
            "holiday_name": holiday_name if is_holiday else "",
            "day_of_week": current.weekday(),
            "is_weekend": is_weekend,
            "is_school_vacation": is_school_vac,
            "month": current.month,
            "week_of_year": current.isocalendar()[1],
        })
        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info("Generated %d calendar rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_calendar(df: pd.DataFrame) -> None:
    """Upload calendar data as a single Parquet file (idempotent overwrite)."""
    key = f"{PREFIX}calendar.parquet"
    upload_parquet_to_s3(df, BUCKET, key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Belgian calendar data and store in MinIO bronze layer.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="First year to generate (default: 2020).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Last year to generate (default: 2026).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(
        "Calendar generation: %d to %d", args.start_year, args.end_year,
    )

    df = generate_calendar(args.start_year, args.end_year)
    upload_calendar(df)

    logger.info("Calendar generation finished.")


if __name__ == "__main__":
    main()
