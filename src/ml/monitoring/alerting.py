"""Alerting module for the Belgian Energy Demand Forecasting system.

All thresholds are read from ``configs/monitoring/thresholds.yaml`` --
nothing is hardcoded.

Alert levels:
    CRITICAL - MAPE exceeds critical threshold or pipeline failure.
    WARNING  - MAPE exceeds warning threshold or drift PSI exceeds warning.
    INFO     - No issues detected.

Supports optional webhook URL for external notifications.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.shared.config import load_config

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).resolve().parents[3] / "configs"


def _load_alerting_config() -> dict[str, Any]:
    """Read all alerting-related thresholds from the monitoring config.

    Returns:
        Dict with ``mape_warning_threshold``, ``mape_critical_threshold``,
        ``psi_warning_threshold``, and ``webhook_url``.
    """
    cfg = load_config(_CONFIGS_DIR / "monitoring" / "thresholds.yaml")
    perf: dict[str, Any] = cfg.get("performance", {})
    drift: dict[str, Any] = cfg.get("drift", {})
    alerting: dict[str, Any] = cfg.get("alerting", {})
    return {
        "mape_warning_threshold": float(
            perf.get("mape_warning_threshold", 0.07)
        ),
        "mape_critical_threshold": float(
            perf.get("mape_critical_threshold", 0.10)
        ),
        "psi_warning_threshold": float(
            drift.get("psi_warning_threshold", 0.2)
        ),
        "webhook_url": alerting.get("webhook_url"),
    }


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


class AlertLevel(Enum):
    """Severity levels for monitoring alerts."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert:
    """Represents a single monitoring alert.

    Args:
        level: Severity of the alert.
        source: Subsystem that produced the alert (e.g. ``"monitoring"``).
        message: Human-readable description.
        details: Arbitrary key-value metadata.
        title: Short summary title.
        timestamp: When the alert was created (defaults to now).
    """

    def __init__(
        self,
        level: AlertLevel,
        source: str,
        message: str,
        details: dict[str, Any] | None = None,
        title: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        self.level: AlertLevel = level
        self.source: str = source
        self.message: str = message
        self.details: dict[str, Any] = details or {}
        self.title: str = title
        self.timestamp: datetime = timestamp or datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the alert to a plain dictionary.

        Returns:
            Dict suitable for JSON serialization.
        """
        return {
            "level": self.level.value,
            "title": self.title,
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------


def classify_alert(
    mape: float | None = None,
    max_psi: float | None = None,
    pipeline_failure: bool = False,
    mape_warning_threshold: float | None = None,
    mape_critical_threshold: float | None = None,
    psi_warning_threshold: float | None = None,
) -> AlertLevel:
    """Classify the appropriate alert level based on metric values.

    Thresholds that default to ``None`` are loaded from the monitoring
    config file.

    Args:
        mape: Current MAPE as a fraction (e.g. 0.08 for 8 %).
        max_psi: Maximum PSI score across all monitored features.
        pipeline_failure: Whether a pipeline failure occurred.
        mape_warning_threshold: MAPE fraction threshold for WARNING.
        mape_critical_threshold: MAPE fraction threshold for CRITICAL.
        psi_warning_threshold: PSI threshold for WARNING.

    Returns:
        The highest applicable ``AlertLevel``.
    """
    cfg: dict[str, Any] = _load_alerting_config()
    if mape_warning_threshold is None:
        mape_warning_threshold = cfg["mape_warning_threshold"]
    if mape_critical_threshold is None:
        mape_critical_threshold = cfg["mape_critical_threshold"]
    if psi_warning_threshold is None:
        psi_warning_threshold = cfg["psi_warning_threshold"]

    if pipeline_failure:
        return AlertLevel.CRITICAL

    if mape is not None and mape > mape_critical_threshold:
        return AlertLevel.CRITICAL

    if mape is not None and mape > mape_warning_threshold:
        return AlertLevel.WARNING

    if max_psi is not None and max_psi > psi_warning_threshold:
        return AlertLevel.WARNING

    return AlertLevel.INFO


# Backward-compatible alias
determine_alert_level = classify_alert


# ------------------------------------------------------------------
# Alert dispatching
# ------------------------------------------------------------------


def _send_webhook(alert: Alert, webhook_url: str) -> bool:
    """Send an alert payload to an external webhook URL.

    Args:
        alert: The alert to send.
        webhook_url: Destination URL for the webhook POST request.

    Returns:
        ``True`` if the webhook was sent successfully, ``False`` otherwise.
    """
    payload: bytes = json.dumps(alert.to_dict()).encode("utf-8")
    req = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            logger.info(
                "Webhook sent to %s (status %d)", webhook_url, resp.status
            )
            return True
    except (URLError, OSError) as exc:
        logger.error("Failed to send webhook to %s: %s", webhook_url, exc)
        return False


def send_alert(
    alert: Alert,
    webhook_url: str | None = None,
) -> None:
    """Dispatch an alert: always log, optionally send webhook.

    INFO-level alerts are logged only.  WARNING and CRITICAL alerts are
    also sent to the webhook if a URL is provided (either via argument
    or from config).

    Args:
        alert: The alert to dispatch.
        webhook_url: Optional webhook URL for external notifications.
            Falls back to the config value if ``None``.
    """
    log_fn = {
        AlertLevel.INFO: logger.info,
        AlertLevel.WARNING: logger.warning,
        AlertLevel.CRITICAL: logger.critical,
    }.get(alert.level, logger.info)

    log_fn(
        "[%s] %s: %s | Details: %s",
        alert.level.value,
        alert.source,
        alert.message,
        alert.details,
    )

    if webhook_url is None:
        webhook_url = _load_alerting_config().get("webhook_url")

    if webhook_url and alert.level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
        _send_webhook(alert, webhook_url)


# Backward-compatible alias
dispatch_alert = send_alert


# ------------------------------------------------------------------
# Convenience: combined check-and-alert
# ------------------------------------------------------------------


def check_and_alert(
    mape: float | None = None,
    max_psi: float | None = None,
    pipeline_failure: bool = False,
    webhook_url: str | None = None,
) -> Alert:
    """Classify metrics, build an alert, and dispatch it.

    Args:
        mape: Current MAPE as a fraction.
        max_psi: Maximum PSI score across features.
        pipeline_failure: Whether a pipeline failure occurred.
        webhook_url: Optional webhook URL.

    Returns:
        The created ``Alert`` object.
    """
    level: AlertLevel = classify_alert(
        mape=mape, max_psi=max_psi, pipeline_failure=pipeline_failure
    )

    messages: list[str] = []
    if pipeline_failure:
        messages.append("Data pipeline failure detected")
    if mape is not None:
        messages.append(f"Current MAPE: {mape:.4f}")
    if max_psi is not None:
        messages.append(f"Max feature PSI: {max_psi:.4f}")

    alert = Alert(
        level=level,
        source="monitoring",
        message=" | ".join(messages) if messages else "No issues detected",
        details={
            "mape": mape,
            "max_psi": max_psi,
            "pipeline_failure": pipeline_failure,
        },
    )

    send_alert(alert, webhook_url=webhook_url)

    # Trigger automated retraining on critical alerts
    trigger_retrain_if_critical(alert)

    return alert


def trigger_retrain_if_critical(alert: Alert) -> bool:
    """Trigger automated retraining when a CRITICAL alert is raised.

    In production this triggers an Airflow DAG or Kubernetes Job.
    For local development it logs a warning with instructions.

    Args:
        alert: The alert to evaluate.

    Returns:
        ``True`` if retrain was triggered, ``False`` otherwise.
    """
    if alert.level != AlertLevel.CRITICAL:
        return False

    logger.critical(
        "CRITICAL alert triggered automated retrain: %s", alert.message
    )

    # Try to trigger via Airflow API (if available)
    import os

    airflow_url: str | None = os.environ.get("AIRFLOW_API_URL")
    if airflow_url:
        try:
            req = Request(
                f"{airflow_url}/api/v1/dags/dag_retrain/dagRuns",
                data=json.dumps(
                    {"conf": {"trigger": "drift_alert", "reason": alert.message}}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(req, timeout=10)
            logger.info("Retrain DAG triggered via Airflow API")
            return True
        except Exception as exc:
            logger.warning("Failed to trigger Airflow retrain: %s", exc)

    # Fallback: log instructions
    logger.warning(
        "Automated retrain requested but no Airflow API configured. "
        "Run manually: python run.py train"
    )
    return False
