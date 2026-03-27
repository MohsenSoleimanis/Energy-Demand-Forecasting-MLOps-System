"""
Alerting module for the Belgian Energy Demand Forecasting system.

MON-003

Alert levels:
    CRITICAL - MAPE > 10% or pipeline failure: log + could trigger retrain
    WARNING  - MAPE > 7% or drift PSI > 0.2: log + notification
    INFO     - Minor drift: log only

Supports optional webhook URL for external notifications.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (as fractions, matching test expectations)
# ---------------------------------------------------------------------------
MAPE_WARNING_THRESHOLD = 0.07   # 7%
MAPE_CRITICAL_THRESHOLD = 0.10  # 10%
PSI_WARNING_THRESHOLD = 0.2


class AlertLevel(str, Enum):
    """Severity levels for monitoring alerts."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert:
    """Represents a single monitoring alert."""

    def __init__(
        self,
        level: AlertLevel,
        source: str,
        message: str,
        details: Optional[dict] = None,
        title: str = "",
        timestamp: Optional[datetime] = None,
    ):
        self.level = level
        self.source = source
        self.message = message
        self.details = details or {}
        self.title = title
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "title": self.title,
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


def classify_alert(
    mape: Optional[float] = None,
    max_psi: Optional[float] = None,
    pipeline_failure: bool = False,
    mape_warning_threshold: float = MAPE_WARNING_THRESHOLD,
    mape_critical_threshold: float = MAPE_CRITICAL_THRESHOLD,
    psi_warning_threshold: float = PSI_WARNING_THRESHOLD,
) -> AlertLevel:
    """
    Classify the appropriate alert level based on metric values.

    Args:
        mape: Current MAPE as a fraction (e.g. 0.08 for 8%).
        max_psi: Maximum PSI score across all monitored features.
        pipeline_failure: Whether a pipeline failure occurred.
        mape_warning_threshold: MAPE fraction threshold for WARNING.
        mape_critical_threshold: MAPE fraction threshold for CRITICAL.
        psi_warning_threshold: PSI threshold for WARNING.

    Returns:
        The highest applicable AlertLevel.
    """
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


# ---------------------------------------------------------------------------
# Alert dispatching
# ---------------------------------------------------------------------------
def _send_webhook(alert: Alert, webhook_url: str) -> bool:
    """
    Send an alert payload to an external webhook URL.

    Args:
        alert: The alert to send.
        webhook_url: Destination URL for the webhook POST request.

    Returns:
        True if the webhook was sent successfully, False otherwise.
    """
    payload = json.dumps(alert.to_dict()).encode("utf-8")
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
    except (URLError, OSError) as e:
        logger.error("Failed to send webhook to %s: %s", webhook_url, e)
        return False


def send_alert(
    alert: Alert,
    webhook_url: Optional[str] = None,
) -> None:
    """
    Send/dispatch an alert: always log, optionally send webhook.

    INFO-level alerts are logged only. WARNING and CRITICAL alerts are
    also sent to the webhook if a URL is provided.

    Args:
        alert: The alert to dispatch.
        webhook_url: Optional webhook URL for external notifications.
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

    if webhook_url and alert.level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
        _send_webhook(alert, webhook_url)


# Backward-compatible alias
dispatch_alert = send_alert


# ---------------------------------------------------------------------------
# Convenience: combined check-and-alert
# ---------------------------------------------------------------------------
def check_and_alert(
    mape: Optional[float] = None,
    max_psi: Optional[float] = None,
    pipeline_failure: bool = False,
    webhook_url: Optional[str] = None,
) -> Alert:
    """
    Classify metrics, build an alert, and dispatch it.

    Args:
        mape: Current MAPE as a fraction.
        max_psi: Maximum PSI score across features.
        pipeline_failure: Whether a pipeline failure occurred.
        webhook_url: Optional webhook URL.

    Returns:
        The created Alert object.
    """
    level = classify_alert(
        mape=mape, max_psi=max_psi, pipeline_failure=pipeline_failure
    )

    messages = []
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
    return alert
