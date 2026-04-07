"""Prometheus metric definitions and helper functions.

All metrics are created at module level so they are registered once with the
default ``prometheus_client`` registry.  Helper functions provide a convenient
high-level API for the serving and monitoring layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prometheus_client import Counter, Gauge, Histogram, Info

if TYPE_CHECKING:
    pass  # reserved for future type-only imports

# ── HTTP / middleware metrics ─────────────────────────────────────────────────

REQUEST_COUNT: Counter = Counter(
    "http_requests_total",
    "Total number of HTTP requests.",
    labelnames=["method", "path", "status_code"],
)

REQUEST_LATENCY: Histogram = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ── Prediction metrics ────────────────────────────────────────────────────────

PREDICTION_LATENCY: Histogram = Histogram(
    "prediction_latency_seconds",
    "Time taken for a single model prediction.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTION_COUNT: Counter = Counter(
    "prediction_total",
    "Total number of predictions served.",
    labelnames=["model_version", "status"],
)

PREDICTION_VALUE: Histogram = Histogram(
    "prediction_value_kwh",
    "Distribution of predicted energy demand values (kWh).",
    buckets=(0, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 2000),
)

# ── Drift metrics ────────────────────────────────────────────────────────────

DRIFT_SCORE: Gauge = Gauge(
    "drift_score",
    "Per-feature drift score (0–1). Values near 1 indicate strong drift.",
    labelnames=["feature"],
)

# ── Model metrics ────────────────────────────────────────────────────────────

MODEL_LOADED: Gauge = Gauge(
    "model_loaded",
    "Whether a model is currently loaded (1) or not (0).",
    labelnames=["model_name", "stage"],
)

ACTIVE_MODEL_VERSION: Info = Info(
    "active_model",
    "Metadata about the active prediction model.",
)


# ── Helper functions ──────────────────────────────────────────────────────────


def track_prediction_metrics(
    latency: float,
    value: float,
    model_version: str,
    status: str = "success",
) -> None:
    """Record metrics for a single prediction call.

    Parameters
    ----------
    latency:
        Wall-clock time for the prediction in **seconds**.
    value:
        The predicted kWh value.
    model_version:
        Version string of the model that produced the prediction.
    status:
        ``"success"`` or ``"error"``.
    """
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_COUNT.labels(model_version=model_version, status=status).inc()
    if status == "success":
        PREDICTION_VALUE.observe(value)


def update_drift_metrics(drift_report: Any) -> None:
    """Push per-feature drift scores from a :class:`DriftReport` to Prometheus.

    Parameters
    ----------
    drift_report:
        A ``DriftReport`` instance (see ``monitoring.drift``).
    """
    for feature_name, result in drift_report.feature_drift.items():
        DRIFT_SCORE.labels(feature=feature_name).set(result.drift_score)


def update_model_info(model_name: str, version: str, stage: str) -> None:
    """Update the ``active_model`` Info metric and the ``model_loaded`` gauge.

    Parameters
    ----------
    model_name:
        Registered model name (e.g. ``"energy-demand-forecaster"``).
    version:
        Model version string.
    stage:
        Model stage (``"Production"``, ``"Staging"``, etc.).
    """
    ACTIVE_MODEL_VERSION.info(
        {"model_name": model_name, "version": version, "stage": stage}
    )
    MODEL_LOADED.labels(model_name=model_name, stage=stage).set(1)
