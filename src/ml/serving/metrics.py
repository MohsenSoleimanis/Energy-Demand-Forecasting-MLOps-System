"""
Prometheus metrics for the Energy Demand Forecasting API.

SERVE-004
"""

from prometheus_client import Counter, Gauge, Histogram

# Total prediction requests, labeled by endpoint and status
PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)

# Prediction latency in seconds, labeled by endpoint
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Current model version info gauge
MODEL_VERSION_GAUGE = Gauge(
    "model_version_info",
    "Current model version",
    ["model_name", "version"],
)

# Distribution of predicted load values in MW
PREDICTION_VALUE = Histogram(
    "predicted_load_mw",
    "Distribution of predicted values",
    buckets=[4000, 6000, 8000, 10000, 12000, 14000, 16000],
)
