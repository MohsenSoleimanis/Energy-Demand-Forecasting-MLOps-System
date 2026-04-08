"""Model performance tracking and degradation detection."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from energy_forecast.evaluation.metrics import MetricsCalculator


@dataclass
class DegradationResult:
    """Result of degradation detection."""
    is_degraded: bool
    metric_deltas: dict[str, float]
    alert_message: str


class PerformanceMonitor:
    """Tracks model performance over time and detects degradation."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.metrics_to_track = self.config.get("metrics", ["mae", "rmse", "mape", "r2"])
        self.degradation_threshold = self.config.get("degradation_threshold", 0.15)
        self.rolling_window = self.config.get("rolling_window", 7)
        self._history: list[dict] = []

    def track_prediction(self, y_true: np.ndarray, y_pred: np.ndarray,
                         timestamp: Optional[datetime] = None) -> dict[str, float]:
        """Record a prediction batch for monitoring. Returns computed metrics."""
        timestamp = timestamp or datetime.utcnow()
        metrics = MetricsCalculator.compute_all(y_true, y_pred)
        record = {"timestamp": timestamp, **metrics}
        self._history.append(record)
        return metrics

    def compute_metrics(self, window_days: Optional[int] = None) -> dict[str, float]:
        """Compute rolling metrics over a time window."""
        if not self._history:
            return {}

        window_days = window_days or self.rolling_window
        df = pd.DataFrame(self._history)
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        recent = df[df["timestamp"] >= cutoff]

        if recent.empty:
            recent = df

        result = {}
        for metric in self.metrics_to_track:
            if metric in recent.columns:
                result[f"{metric}_mean"] = float(recent[metric].mean())
                result[f"{metric}_std"] = float(recent[metric].std())
                result[f"{metric}_min"] = float(recent[metric].min())
                result[f"{metric}_max"] = float(recent[metric].max())
        return result

    def detect_degradation(self, baseline_metrics: dict[str, float],
                           current_metrics: Optional[dict[str, float]] = None,
                           threshold: Optional[float] = None) -> DegradationResult:
        """Detect if model performance has degraded from baseline."""
        threshold = threshold or self.degradation_threshold
        if current_metrics is None:
            current_metrics = self.compute_metrics()

        metric_deltas = {}
        is_degraded = False
        alerts = []

        for metric in self.metrics_to_track:
            baseline_key = metric
            current_key = f"{metric}_mean"

            baseline_val = baseline_metrics.get(baseline_key)
            current_val = current_metrics.get(current_key)

            if baseline_val is None or current_val is None:
                continue

            if baseline_val == 0:
                delta = 0.0
            else:
                # For r2, higher is better; for others, lower is better
                if metric == "r2":
                    delta = (baseline_val - current_val) / abs(baseline_val)
                else:
                    delta = (current_val - baseline_val) / abs(baseline_val)

            metric_deltas[metric] = delta

            if delta > threshold:
                is_degraded = True
                alerts.append(
                    f"{metric} degraded by {delta:.1%} "
                    f"(baseline: {baseline_val:.4f}, current: {current_val:.4f})"
                )

        alert_message = "; ".join(alerts) if alerts else "Model performance is within acceptable bounds."

        return DegradationResult(
            is_degraded=is_degraded,
            metric_deltas=metric_deltas,
            alert_message=alert_message,
        )

    def get_performance_history(self) -> pd.DataFrame:
        """Return all tracked metrics as a DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history).set_index("timestamp")

    def reset(self) -> None:
        """Clear all tracked history."""
        self._history = []
