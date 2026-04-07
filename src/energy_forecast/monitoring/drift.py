"""Data drift detection using statistical tests.

This module provides a pure NumPy / SciPy implementation of drift detection --
**no dependency on evidently** or other heavy monitoring libraries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class FeatureDriftResult:
    """Drift test outcome for a single feature."""

    feature_name: str
    is_drifted: bool
    drift_score: float
    test_name: str
    p_value: float
    threshold: float


@dataclass
class DriftReport:
    """Aggregated drift report across all monitored features."""

    is_drifted: bool
    feature_drift: dict[str, FeatureDriftResult]
    dataset_drift_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Main class ────────────────────────────────────────────────────────────────


class DriftDetector:
    """Detect data drift between a reference dataset and current (production) data.

    Parameters
    ----------
    reference_data:
        A :class:`~pandas.DataFrame` representing the training / baseline
        distribution.
    config:
        Dictionary of configuration values.  Expected keys (all optional):

        * ``features_to_monitor`` – list of column names to check.
        * ``thresholds.feature_drift`` – p-value / PSI threshold per feature
          (default ``0.3``).
        * ``thresholds.dataset_drift`` – fraction of features that must be
          drifted for the whole dataset to be flagged (default ``0.5``).
    """

    def __init__(self, reference_data: pd.DataFrame, config: dict[str, Any] | None = None) -> None:
        self._reference = reference_data.copy()
        self._config = config or {}

        self._features: list[str] = self._config.get(
            "features_to_monitor",
            [c for c in reference_data.columns if reference_data[c].dtype.kind in ("f", "i", "u")],
        )
        thresholds = self._config.get("thresholds", {})
        self._feature_threshold: float = float(thresholds.get("feature_drift", 0.3))
        self._dataset_threshold: float = float(thresholds.get("dataset_drift", 0.5))

    # ── Public API ────────────────────────────────────────────────────────

    def detect_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Run drift tests on *current_data* and return a :class:`DriftReport`."""
        feature_results: dict[str, FeatureDriftResult] = {}

        for feat in self._features:
            if feat not in self._reference.columns or feat not in current_data.columns:
                logger.warning("Feature '%s' missing from one of the datasets – skipped.", feat)
                continue
            ref_vals = self._reference[feat].dropna().values.astype(float)
            cur_vals = current_data[feat].dropna().values.astype(float)

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                logger.warning("Feature '%s' has no non-null values – skipped.", feat)
                continue

            result = self._test_feature(feat, ref_vals, cur_vals)
            feature_results[feat] = result

        # Dataset-level drift decision
        if feature_results:
            n_drifted = sum(1 for r in feature_results.values() if r.is_drifted)
            dataset_score = n_drifted / len(feature_results)
        else:
            dataset_score = 0.0

        is_drifted = dataset_score >= self._dataset_threshold

        return DriftReport(
            is_drifted=is_drifted,
            feature_drift=feature_results,
            dataset_drift_score=round(dataset_score, 4),
        )

    def generate_report(self, current_data: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return a JSON-serialisable summary suitable for logging or alerting.

        If *current_data* is ``None`` the report only contains reference
        statistics (useful as a baseline snapshot).
        """
        if current_data is None:
            return {
                "type": "drift_baseline",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reference_rows": len(self._reference),
                "features_monitored": self._features,
            }

        report = self.detect_drift(current_data)
        feature_summaries: dict[str, Any] = {}
        for name, result in report.feature_drift.items():
            feature_summaries[name] = {
                "is_drifted": result.is_drifted,
                "drift_score": result.drift_score,
                "test": result.test_name,
                "p_value": result.p_value,
                "threshold": result.threshold,
            }

        return {
            "type": "drift_report",
            "timestamp": report.timestamp.isoformat(),
            "is_drifted": report.is_drifted,
            "dataset_drift_score": report.dataset_drift_score,
            "features": feature_summaries,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _test_feature(
        self,
        name: str,
        ref: np.ndarray,
        cur: np.ndarray,
    ) -> FeatureDriftResult:
        """Apply the KS test **and** PSI, returning the combined result."""
        # Kolmogorov–Smirnov test
        ks_stat, ks_p = stats.ks_2samp(ref, cur)

        # Population Stability Index
        psi_value = self._compute_psi(ref, cur)

        # We flag drift if KS p-value < 0.05 **or** PSI exceeds the threshold.
        is_drifted = (ks_p < 0.05) or (psi_value > self._feature_threshold)
        drift_score = round(float(psi_value), 6)

        return FeatureDriftResult(
            feature_name=name,
            is_drifted=is_drifted,
            drift_score=drift_score,
            test_name="ks_test+psi",
            p_value=round(float(ks_p), 6),
            threshold=self._feature_threshold,
        )

    @staticmethod
    def _compute_psi(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute the Population Stability Index between two distributions.

        PSI = Σ (P_i - Q_i) * ln(P_i / Q_i)

        where P and Q are the bin-proportions of the reference and current
        distributions respectively.
        """
        # Create bins from reference distribution
        epsilon = 1e-4
        breakpoints = np.linspace(
            min(reference.min(), current.min()) - epsilon,
            max(reference.max(), current.max()) + epsilon,
            n_bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
        cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

        # Convert to proportions, avoiding zeros
        ref_pct = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
        cur_pct = np.clip(cur_counts / cur_counts.sum(), epsilon, None)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi
