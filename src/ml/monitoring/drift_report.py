"""
Data drift detection using Evidently AI.

MON-001

Compares a reference dataset (training period) against a current window
(last 7 days) and generates an HTML drift report stored in MinIO.
"""

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

# Features monitored for drift
NUMERIC_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "price_eur_mwh",
]

DEFAULT_PSI_THRESHOLD = 0.2


def load_reference_data(
    path: str,
    days: int = 30,
) -> pd.DataFrame:
    """
    Load reference data for drift comparison.

    Args:
        path: Path (local or MinIO/S3) to the parquet feature store.
        days: Number of days from the start of the test period to use
              as reference.

    Returns:
        DataFrame with the reference window.
    """
    df = pd.read_parquet(path)
    if "timestamp_brussels" in df.columns:
        df["timestamp_brussels"] = pd.to_datetime(df["timestamp_brussels"])
        start = df["timestamp_brussels"].min()
        end = start + timedelta(days=days)
        df = df[df["timestamp_brussels"] < end]
    logger.info("Loaded reference data: %d rows", len(df))
    return df


def load_current_data(
    path: str,
    days: int = 7,
) -> pd.DataFrame:
    """
    Load the most recent data window for drift comparison.

    Args:
        path: Path (local or MinIO/S3) to the parquet feature store.
        days: Number of recent days to include.

    Returns:
        DataFrame with the current window.
    """
    df = pd.read_parquet(path)
    if "timestamp_brussels" in df.columns:
        df["timestamp_brussels"] = pd.to_datetime(df["timestamp_brussels"])
        cutoff = df["timestamp_brussels"].max() - timedelta(days=days)
        df = df[df["timestamp_brussels"] >= cutoff]
    logger.info("Loaded current data: %d rows", len(df))
    return df


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> dict:
    """
    Run Evidently DataDriftPreset and persist the HTML report.

    Args:
        reference: Reference (training) DataFrame.
        current: Current (recent) DataFrame.
        output_path: Where to save the HTML report.
        psi_threshold: PSI threshold above which a feature is considered
                       drifted.

    Returns:
        Dictionary with per-feature drift scores and an ``alert`` flag.
    """
    column_mapping = ColumnMapping(
        numerical_features=[
            f for f in NUMERIC_FEATURES if f in reference.columns
        ],
        target=None,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    # Persist HTML
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info("Drift report saved to %s", output_path)

    # Extract per-feature drift scores
    result = report.as_dict()
    drift_scores: dict[str, float] = {}
    alert = False

    metrics = result.get("metrics", [])
    for metric in metrics:
        metric_result = metric.get("result", {})
        drift_by_columns = metric_result.get("drift_by_columns", {})
        for col_name, col_info in drift_by_columns.items():
            score = col_info.get("drift_score", 0.0)
            drift_scores[col_name] = score
            if score > psi_threshold:
                logger.warning(
                    "Drift detected for %s: PSI=%.4f (threshold=%.4f)",
                    col_name,
                    score,
                    psi_threshold,
                )
                alert = True

    return {
        "drift_scores": drift_scores,
        "alert": alert,
        "report_path": output_path,
    }


def run_drift_check(
    feature_store_path: str,
    reference_days: int = 30,
    current_days: int = 7,
    output_path: str = "reports/drift_report.html",
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    alert_callback: callable | None = None,
) -> dict:
    """
    End-to-end drift check: load data, generate report, optionally alert.

    Args:
        feature_store_path: Path to the feature store parquet file.
        reference_days: Days for the reference window.
        current_days: Days for the current window.
        output_path: Where to save the HTML report.
        psi_threshold: PSI threshold for drift alerting.
        alert_callback: Optional callable invoked when drift is detected.
                        Receives the drift result dict.

    Returns:
        Drift result dictionary.
    """
    reference = load_reference_data(feature_store_path, days=reference_days)
    current = load_current_data(feature_store_path, days=current_days)

    result = generate_drift_report(
        reference,
        current,
        output_path=output_path,
        psi_threshold=psi_threshold,
    )

    if result["alert"] and alert_callback is not None:
        alert_callback(result)

    return result
