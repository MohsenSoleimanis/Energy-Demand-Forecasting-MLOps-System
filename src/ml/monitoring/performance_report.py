"""
Model performance monitoring using Evidently AI.

MON-002

Compares recent prediction accuracy (MAPE) against the registered
production model baseline and generates a regression performance report.
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import RegressionPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

DEFAULT_MAPE_WARNING = 7.0
DEFAULT_MAPE_CRITICAL = 10.0


def compute_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        MAPE as a percentage (e.g. 5.3 means 5.3%).
    """
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def load_monitoring_set(
    path: str,
    days: int = 7,
) -> pd.DataFrame:
    """
    Load a dataset containing both predictions and actuals.

    Args:
        path: Path to the monitoring parquet file.
        days: Number of recent days to evaluate.

    Returns:
        Filtered DataFrame.
    """
    df = pd.read_parquet(path)
    if "timestamp_brussels" in df.columns:
        df["timestamp_brussels"] = pd.to_datetime(df["timestamp_brussels"])
        cutoff = df["timestamp_brussels"].max() - timedelta(days=days)
        df = df[df["timestamp_brussels"] >= cutoff]
    logger.info("Loaded monitoring set: %d rows", len(df))
    return df


def generate_performance_report(
    df: pd.DataFrame,
    target_col: str = "total_load_actual_mw",
    prediction_col: str = "predicted_load_mw",
    output_path: str = "reports/performance_report.html",
    registered_mape: Optional[float] = None,
    mape_warning: float = DEFAULT_MAPE_WARNING,
    mape_critical: float = DEFAULT_MAPE_CRITICAL,
) -> dict:
    """
    Compute performance metrics and generate an Evidently regression report.

    Args:
        df: DataFrame with both actual and predicted columns.
        target_col: Name of the column containing actual load values.
        prediction_col: Name of the column containing predicted load values.
        output_path: Where to save the HTML report.
        registered_mape: The production model's registered MAPE for comparison.
        mape_warning: MAPE threshold for WARNING alert.
        mape_critical: MAPE threshold for CRITICAL alert.

    Returns:
        Dictionary with computed MAPE, alert level, and report path.
    """
    if target_col not in df.columns or prediction_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{target_col}' and '{prediction_col}' columns."
        )

    current_mape = compute_mape(df[target_col], df[prediction_col])
    logger.info("Current MAPE: %.2f%%", current_mape)

    # Determine alert level
    alert_level = "INFO"
    if current_mape > mape_critical:
        alert_level = "CRITICAL"
    elif current_mape > mape_warning:
        alert_level = "WARNING"

    if registered_mape is not None:
        degradation = current_mape - registered_mape
        logger.info(
            "MAPE degradation vs registered: %.2f%% (registered: %.2f%%)",
            degradation,
            registered_mape,
        )

    # Evidently regression report
    column_mapping = ColumnMapping(
        target=target_col,
        prediction=prediction_col,
    )

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=None, current_data=df, column_mapping=column_mapping)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info("Performance report saved to %s", output_path)

    return {
        "current_mape": current_mape,
        "registered_mape": registered_mape,
        "alert_level": alert_level,
        "report_path": output_path,
    }


def run_performance_check(
    monitoring_set_path: str,
    days: int = 7,
    target_col: str = "total_load_actual_mw",
    prediction_col: str = "predicted_load_mw",
    output_path: str = "reports/performance_report.html",
    registered_mape: Optional[float] = None,
    mape_warning: float = DEFAULT_MAPE_WARNING,
    mape_critical: float = DEFAULT_MAPE_CRITICAL,
    alert_callback: Optional[callable] = None,
) -> dict:
    """
    End-to-end performance check: load data, compute metrics, alert.

    Args:
        monitoring_set_path: Path to monitoring data parquet file.
        days: Number of recent days to evaluate.
        target_col: Actual values column name.
        prediction_col: Predicted values column name.
        output_path: Where to save the HTML report.
        registered_mape: Production model's registered MAPE.
        mape_warning: MAPE threshold for WARNING alert.
        mape_critical: MAPE threshold for CRITICAL alert.
        alert_callback: Optional callable invoked when alert_level is not INFO.

    Returns:
        Performance result dictionary.
    """
    df = load_monitoring_set(monitoring_set_path, days=days)

    result = generate_performance_report(
        df,
        target_col=target_col,
        prediction_col=prediction_col,
        output_path=output_path,
        registered_mape=registered_mape,
        mape_warning=mape_warning,
        mape_critical=mape_critical,
    )

    if result["alert_level"] != "INFO" and alert_callback is not None:
        alert_callback(result)

    return result
