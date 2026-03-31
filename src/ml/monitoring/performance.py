"""Model performance monitoring using Evidently AI.

Compares recent prediction accuracy (MAPE) against configurable
thresholds and generates a regression performance report.

Thresholds are read from ``configs/monitoring/thresholds.yaml`` and
MAPE is computed via :func:`src.shared.metrics.compute_mape` -- no
local reimplementation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import RegressionPreset
from evidently.report import Report

from src.shared.config import load_config
from src.shared.metrics import compute_mape

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).resolve().parents[3] / "configs"


def _load_performance_thresholds() -> dict[str, Any]:
    """Read performance-related thresholds from the monitoring config.

    Returns:
        Dict with ``mape_warning_threshold``, ``mape_critical_threshold``,
        and ``evaluation_window_days``.
    """
    cfg = load_config(_CONFIGS_DIR / "monitoring" / "thresholds.yaml")
    perf_cfg: dict[str, Any] = cfg.get("performance", {})
    return {
        "mape_warning_threshold": float(
            perf_cfg.get("mape_warning_threshold", 0.07)
        ),
        "mape_critical_threshold": float(
            perf_cfg.get("mape_critical_threshold", 0.10)
        ),
        "evaluation_window_days": int(
            perf_cfg.get("evaluation_window_days", 7)
        ),
    }


# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------


def load_monitoring_data(
    path: str, days: int | None = None
) -> pd.DataFrame:
    """Load a dataset containing both predictions and actuals.

    Args:
        path: Path to the monitoring parquet file.
        days: Number of recent days to evaluate.  Defaults to the
            ``evaluation_window_days`` value in the monitoring config.

    Returns:
        Filtered DataFrame.
    """
    if days is None:
        days = _load_performance_thresholds()["evaluation_window_days"]

    df: pd.DataFrame = pd.read_parquet(path)
    if "timestamp_brussels" in df.columns:
        df["timestamp_brussels"] = pd.to_datetime(df["timestamp_brussels"])
        cutoff = df["timestamp_brussels"].max() - timedelta(days=days)
        df = df[df["timestamp_brussels"] >= cutoff]
    logger.info("Loaded monitoring data: %d rows", len(df))
    return df


def generate_performance_report(
    df: pd.DataFrame,
    target_col: str = "total_load_actual_mw",
    prediction_col: str = "predicted_load_mw",
    output_path: str = "reports/performance_report.html",
    registered_mape: float | None = None,
    mape_warning: float | None = None,
    mape_critical: float | None = None,
) -> dict[str, Any]:
    """Compute performance metrics and generate an Evidently regression report.

    Args:
        df: DataFrame with both actual and predicted columns.
        target_col: Name of the column containing actual load values.
        prediction_col: Name of the column containing predicted load values.
        output_path: Where to save the HTML report.
        registered_mape: The production model's registered MAPE for comparison.
        mape_warning: MAPE threshold (as a fraction) for WARNING alert.
            Defaults to the value in the monitoring config.
        mape_critical: MAPE threshold (as a fraction) for CRITICAL alert.
            Defaults to the value in the monitoring config.

    Returns:
        Dictionary with ``current_mape``, ``registered_mape``,
        ``alert_level``, and ``report_path``.

    Raises:
        ValueError: If required columns are missing from *df*.
    """
    thresholds = _load_performance_thresholds()
    if mape_warning is None:
        mape_warning = thresholds["mape_warning_threshold"]
    if mape_critical is None:
        mape_critical = thresholds["mape_critical_threshold"]

    if target_col not in df.columns or prediction_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{target_col}' and '{prediction_col}' columns."
        )

    current_mape: float = compute_mape(
        df[target_col].values, df[prediction_col].values
    )
    logger.info("Current MAPE: %.4f (%.2f%%)", current_mape, current_mape * 100)

    # Determine alert level
    alert_level: str = "INFO"
    if current_mape > mape_critical:
        alert_level = "CRITICAL"
    elif current_mape > mape_warning:
        alert_level = "WARNING"

    if registered_mape is not None:
        degradation: float = current_mape - registered_mape
        logger.info(
            "MAPE degradation vs registered: %.4f (registered: %.4f)",
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
    days: int | None = None,
    target_col: str = "total_load_actual_mw",
    prediction_col: str = "predicted_load_mw",
    output_path: str = "reports/performance_report.html",
    registered_mape: float | None = None,
    mape_warning: float | None = None,
    mape_critical: float | None = None,
    alert_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """End-to-end performance check: load data, compute metrics, alert.

    All threshold parameters that default to ``None`` are read from
    ``configs/monitoring/thresholds.yaml``.

    Args:
        monitoring_set_path: Path to monitoring data parquet file.
        days: Number of recent days to evaluate.
        target_col: Actual values column name.
        prediction_col: Predicted values column name.
        output_path: Where to save the HTML report.
        registered_mape: Production model's registered MAPE.
        mape_warning: MAPE threshold for WARNING alert.
        mape_critical: MAPE threshold for CRITICAL alert.
        alert_callback: Optional callable invoked when alert_level is
            not ``"INFO"``.

    Returns:
        Performance result dictionary.
    """
    df: pd.DataFrame = load_monitoring_data(monitoring_set_path, days=days)

    result: dict[str, Any] = generate_performance_report(
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
