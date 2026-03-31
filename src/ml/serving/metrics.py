"""Prometheus metrics for the Energy Demand Forecasting API.

Histogram bucket boundaries are read from ``configs/serving/api.yaml``
so they can be tuned without code changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from src.shared.config import load_config

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).resolve().parents[3] / "configs"


def _load_metrics_config() -> dict[str, Any]:
    """Read metric bucket configuration from the serving config file.

    Returns:
        Dict with ``latency_buckets`` and ``prediction_value_buckets``
        lists.
    """
    try:
        cfg = load_config(_CONFIGS_DIR / "serving" / "api.yaml")
        metrics_cfg: dict[str, Any] = cfg.get("metrics", {})
        return {
            "latency_buckets": tuple(
                metrics_cfg.get("latency_buckets", [0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
            ),
            "prediction_value_buckets": tuple(
                metrics_cfg.get(
                    "prediction_value_buckets",
                    [4000, 6000, 8000, 10000, 12000, 14000, 16000],
                )
            ),
        }
    except Exception:
        return {
            "latency_buckets": (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            "prediction_value_buckets": (4000, 6000, 8000, 10000, 12000, 14000, 16000),
        }


_metrics_cfg: dict[str, Any] = _load_metrics_config()

# Total prediction requests, labeled by endpoint and status
PREDICTION_COUNT: Counter = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)

# Prediction latency in seconds, labeled by endpoint
PREDICTION_LATENCY: Histogram = Histogram(
    "prediction_latency_seconds",
    "Prediction latency",
    ["endpoint"],
    buckets=_metrics_cfg["latency_buckets"],
)

# Current model version info gauge
MODEL_VERSION_GAUGE: Gauge = Gauge(
    "model_version_info",
    "Current model version",
    ["model_name", "version"],
)

# Distribution of predicted load values in MW
PREDICTION_VALUE: Histogram = Histogram(
    "predicted_load_mw",
    "Distribution of predicted values",
    buckets=_metrics_cfg["prediction_value_buckets"],
)
