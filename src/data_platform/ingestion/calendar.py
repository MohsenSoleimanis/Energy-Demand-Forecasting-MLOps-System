"""Generate Belgian calendar data including holidays, school vacations, and
temporal features.

Data is stored as a single Parquet file in the MinIO bronze layer. The
school vacation schedule is read from ``configs/data/ingestion.yaml``
(not hardcoded).

All configuration is loaded from ``configs/data/ingestion.yaml``.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import holidays
import pandas as pd

from src.shared.config import load_config
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
# Easter computation
# ---------------------------------------------------------------------------

def _compute_easter(year: int) -> date:
    """Compute Easter Sunday for *year* using the Anonymous Gregorian algorithm.

    This algorithm (also known as the "Meeus/Jones/Butcher" algorithm) computes
    the date of Easter Sunday for any year in the Gregorian calendar.

    Steps:
        1. ``a`` = year mod 19 (the Metonic cycle position).
        2. ``b, c`` = divmod(year, 100) (century and year-within-century).
        3. ``d, e`` = divmod(b, 4) (leap-century correction).
        4. ``f`` = (b + 8) // 25 (additional correction for centuries).
        5. ``g`` = (b - f + 1) // 3 (second century correction).
        6. ``h`` = (19a + b - d - g + 15) mod 30 (preliminary Paschal full moon).
        7. ``i, k`` = divmod(c, 4) (leap-year correction within century).
        8. ``l`` = (32 + 2e + 2i - h - k) mod 7 (day-of-week adjustment).
        9. ``m`` = (a + 11h + 22l) // 451 (correction for certain edge cases).
        10. month = (h + l - 7m + 114) // 31; day = ((h + l - 7m + 114) mod 31) + 1.

    Returns:
        ``datetime.date`` for Easter Sunday of the given year.
    """
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


# ---------------------------------------------------------------------------
# School vacations (config-driven)
# ---------------------------------------------------------------------------

def _school_vacation_ranges(
    year: int, vacation_cfg: dict[str, Any],
) -> list[tuple[date, date]]:
    """Return approximate Flemish school vacation date ranges for *year*.

    The vacation definitions (dates, Easter offsets, etc.) come from the
    ``calendar.school_vacations`` section of the ingestion config.

    Args:
        year: The calendar year.
        vacation_cfg: The ``school_vacations`` dict from config.

    Returns:
        List of ``(start_date, end_date)`` inclusive tuples.
    """
    ranges: list[tuple[date, date]] = []

    # Christmas: Dec 25 of previous year -> Jan 5 of this year
    xmas_start = vacation_cfg["christmas_start"]
    xmas_end = vacation_cfg["christmas_end"]
    ranges.append((
        date(year - 1, xmas_start["month"], xmas_start["day"]),
        date(year, xmas_end["month"], xmas_end["day"]),
    ))

    # Carnival: week around last Friday of February
    last_day_feb = date(year, 3, 1) - timedelta(days=1)
    carnival_friday = last_day_feb
    while carnival_friday.weekday() != 4:  # Friday
        carnival_friday -= timedelta(days=1)
    carnival_monday = carnival_friday - timedelta(days=4)
    ranges.append((carnival_monday, carnival_friday + timedelta(days=2)))

    # Easter: configurable offset and duration relative to Easter Sunday
    easter_sunday = _compute_easter(year)
    easter_offset = vacation_cfg["easter_offset_start_days"]
    easter_duration = vacation_cfg["easter_duration_days"]
    easter_start = easter_sunday + timedelta(days=easter_offset)
    easter_end = easter_start + timedelta(days=easter_duration - 1)
    ranges.append((easter_start, easter_end))

    # Summer
    summer_start = vacation_cfg["summer_start"]
    summer_end = vacation_cfg["summer_end"]
    ranges.append((
        date(year, summer_start["month"], summer_start["day"]),
        date(year, summer_end["month"], summer_end["day"]),
    ))

    # Autumn: week containing Nov 1
    autumn = vacation_cfg["autumn_anchor"]
    nov1 = date(year, autumn["month"], autumn["day"])
    autumn_monday = nov1 - timedelta(days=nov1.weekday())
    autumn_sunday = autumn_monday + timedelta(days=6)
    ranges.append((autumn_monday, autumn_sunday))

    # Christmas starting this year
    ranges.append((
        date(year, xmas_start["month"], xmas_start["day"]),
        date(year, 12, 31),
    ))

    return ranges


def _is_school_vacation(
    d: date, ranges: list[tuple[date, date]],
) -> bool:
    """Check if date *d* falls within any school vacation range.

    Args:
        d: Date to check.
        ranges: Pre-computed vacation ranges for the year.

    Returns:
        ``True`` if *d* is in a vacation period.
    """
    return any(start <= d <= end for start, end in ranges)


# ---------------------------------------------------------------------------
# Calendar generation
# ---------------------------------------------------------------------------

def generate_calendar(
    start_year: int,
    end_year: int,
    vacation_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Generate a daily calendar DataFrame for the given year range.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        vacation_cfg: School vacation definitions from config.

    Returns:
        DataFrame with columns ``date``, ``is_belgian_holiday``,
        ``holiday_name``, ``day_of_week``, ``is_weekend``,
        ``is_school_vacation``, ``month``, ``week_of_year``.
    """
    logger.info("Generating calendar for years %d to %d", start_year, end_year)
    year_range = range(start_year, end_year + 1)
    be_holidays = holidays.Belgium(years=year_range)

    all_vacation_ranges = {
        yr: _school_vacation_ranges(yr, vacation_cfg) for yr in year_range
    }

    rows: list[dict[str, Any]] = []
    current = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)

    while current <= end_date:
        is_holiday = current in be_holidays
        rows.append({
            "date": current.isoformat(),
            "is_belgian_holiday": is_holiday,
            "holiday_name": be_holidays.get(current, "") if is_holiday else "",
            "day_of_week": current.weekday(),
            "is_weekend": current.weekday() >= 5,
            "is_school_vacation": _is_school_vacation(
                current, all_vacation_ranges[current.year],
            ),
            "month": current.month,
            "week_of_year": current.isocalendar()[1],
        })
        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info("Generated %d calendar rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate calendar generation and upload.

    Reads year range and vacation definitions from
    ``configs/data/ingestion.yaml``.
    """
    cfg = _load_ingestion_config()
    cal_cfg = cfg["calendar"]
    storage_cfg = cfg["storage"]

    start_year: int = cal_cfg["start_year"]
    end_year: int = cal_cfg["end_year"]
    vacation_cfg: dict[str, Any] = cal_cfg["school_vacations"]
    bucket: str = storage_cfg["bucket"]
    prefix = f"{storage_cfg['bronze_prefix']}/calendar/"

    df = generate_calendar(start_year, end_year, vacation_cfg)
    key = f"{prefix}calendar.parquet"
    upload_parquet(df, bucket, key)
    logger.info("Uploaded calendar to s3://%s/%s", bucket, key)
    logger.info("Calendar generation finished.")


if __name__ == "__main__":
    main()
