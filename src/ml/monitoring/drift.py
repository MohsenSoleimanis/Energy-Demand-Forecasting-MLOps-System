"""Data drift detection using Evidently AI.

Compares a reference dataset (training period) against a current window
(recent days) and generates an HTML drift report stored locally or in
MinIO.

Feature list is read from ``configs/data/features.yaml`` and
thresholds from ``configs/monitoring/thresholds.yaml`` -- no hardcoded
values.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.shared.config import load_config

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).resolve().parents[3] / "configs"


def _load_monitored_features() -> list[str]:
    """Read the list of monitored numeric features from the features config.

    Returns:
        List of feature column names to monitor for drift.
    """
    cfg = load_config(_CONFIGS_DIR / "data" / "features.yaml")
    return list(cfg.get("monitored_features", []))


def _load_drift_thresholds() -> dict[str, Any]:
    """Read drift-related thresholds from the monitoring config.

    Returns:
        Dict with ``psi_warning_threshold``, ``reference_window_days``,
        and ``current_window_days``.
    """
    cfg = load_config(_CONFIGS_DIR / "monitoring" / "thresholds.yaml")
    drift_cfg: dict[str, Any] = cfg.get("drift", {})
    return {
        "psi_warning_threshold": float(
            drift_cfg.get("psi_warning_threshold", 0.2)
        ),
        "reference_window_days": int(
            drift_cfg.get("reference_window_days", 30)
        ),
        "current_window_days": int(
            drift_cfg.get("current_window_days", 7)
        ),
    }


# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------


def load_reference_data(path: str, days: int | None = None) -> pd.DataFrame:
    """Load reference data for drift comparison.

    Reads the parquet file at *path* and selects the first *days* from
    the earliest ``timestamp_brussels``.

    Args:
        path: Path (local or S3) to the parquet feature store.
        days: Number of days from the start of the data to use as the
            reference window.  Defaults to the value in
            ``configs/monitoring/thresholds.yaml``.

    Returns:
        DataFrame with the reference window.
    """
    if days is None:
        days = _load_drift_thresholds()["reference_window_days"]

    df: pd.DataFrame = pd.read_parquet(path)
    if "timestamp_brussels" in df.columns:
        df["timestamp_brussels"] = pd.to_datetime(df["timestamp_brussels"])
        start = df["timestamp_brussels"].min()
        end = start + timedelta(days=days)
        df = df[df["timestamp_brussels"] < end]
    logger.info("Loaded reference data: %d rows", len(df))
    return df


def load_current_data(path: str, days: int | None = None) -> pd.DataFrame:
    """Load the most recent data window for drift comparison.

    Args:
        path: Path (local or S3) to the parquet feature store.
        days: Number of recent days to include.  Defaults to the value
            in ``configs/monitoring/thresholds.yaml``.

    Returns:
        DataFrame with the current window.
    """
    if days is None:
        days = _load_drift_thresholds()["current_window_days"]

    df: pd.DataFrame = pd.read_parquet(path)
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
    psi_threshold: float | None = None,
) -> dict[str, Any]:
    """Run Evidently DataDriftPreset and persist the HTML report.

    Args:
        reference: Reference (training) DataFrame.
        current: Current (recent) DataFrame.
        output_path: Where to save the HTML report.
        psi_threshold: PSI threshold above which a feature is considered
            drifted.  Defaults to the value in
            ``configs/monitoring/thresholds.yaml``.

    Returns:
        Dictionary with ``drift_scores`` (per-feature), ``alert`` flag,
        and ``report_path``.
    """
    if psi_threshold is None:
        psi_threshold = _load_drift_thresholds()["psi_warning_threshold"]

    monitored: list[str] = _load_monitored_features()
    numeric_features: list[str] = [
        f for f in monitored if f in reference.columns
    ]

    column_mapping = ColumnMapping(
        numerical_features=numeric_features,
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
    alert: bool = False

    metrics_list: list[dict] = result.get("metrics", [])
    for metric in metrics_list:
        metric_result: dict = metric.get("result", {})
        drift_by_columns: dict = metric_result.get("drift_by_columns", {})
        for col_name, col_info in drift_by_columns.items():
            score: float = col_info.get("drift_score", 0.0)
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
    reference_days: int | None = None,
    current_days: int | None = None,
    output_path: str = "reports/drift_report.html",
    psi_threshold: float | None = None,
    alert_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """End-to-end drift check: load data, generate report, optionally alert.

    All parameters that default to ``None`` are read from
    ``configs/monitoring/thresholds.yaml`` at runtime.

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
    thresholds = _load_drift_thresholds()

    if reference_days is None:
        reference_days = thresholds["reference_window_days"]
    if current_days is None:
        current_days = thresholds["current_window_days"]
    if psi_threshold is None:
        psi_threshold = thresholds["psi_warning_threshold"]

    reference: pd.DataFrame = load_reference_data(
        feature_store_path, days=reference_days
    )
    current: pd.DataFrame = load_current_data(
        feature_store_path, days=current_days
    )

    result: dict[str, Any] = generate_drift_report(
        reference,
        current,
        output_path=output_path,
        psi_threshold=psi_threshold,
    )

    if result["alert"] and alert_callback is not None:
        alert_callback(result)

    return result
