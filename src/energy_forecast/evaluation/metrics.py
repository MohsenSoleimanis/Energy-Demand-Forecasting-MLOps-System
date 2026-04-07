"""Metrics calculator for energy demand forecasting models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats


class MetricsCalculator:
    """Collection of static evaluation metrics for regression forecasting.

    All methods accept numpy arrays and return scalar floats (or tuples where
    noted).  The class is stateless and every method is a ``@staticmethod``
    so it can be used without instantiation when convenient.
    """

    @staticmethod
    def mae(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
        """Mean Absolute Error.

        Parameters
        ----------
        y_true:
            Ground-truth target values of shape ``(n,)``.
        y_pred:
            Predicted values of shape ``(n,)``.

        Returns
        -------
        float
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
        """Root Mean Squared Error.

        Parameters
        ----------
        y_true:
            Ground-truth target values.
        y_pred:
            Predicted values.

        Returns
        -------
        float
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
        """Mean Absolute Percentage Error.

        Samples where ``y_true == 0`` are excluded from the computation to
        avoid division-by-zero.  If *all* true values are zero the method
        returns ``0.0``.

        Parameters
        ----------
        y_true:
            Ground-truth target values.
        y_pred:
            Predicted values.

        Returns
        -------
        float
            MAPE expressed as a percentage (e.g. ``5.2`` means 5.2 %).
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        mask = y_true != 0.0
        if not np.any(mask):
            return 0.0

        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

    @staticmethod
    def r2(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
        """Coefficient of determination (R-squared).

        Parameters
        ----------
        y_true:
            Ground-truth target values.
        y_pred:
            Predicted values.

        Returns
        -------
        float
            1.0 for a perfect fit, 0.0 when the model equals the mean
            predictor, negative for worse-than-mean predictions.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def peak_hour_error(
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
        peak_hours: NDArray[np.integer] | list[int] | None = None,
    ) -> float:
        """Mean Absolute Error restricted to peak demand hours.

        Parameters
        ----------
        y_true:
            Ground-truth values.
        y_pred:
            Predicted values.
        peak_hours:
            Boolean mask or integer indices identifying peak-hour samples.
            If ``None``, the top-25 % of ``y_true`` values are treated as
            peak hours automatically.

        Returns
        -------
        float
            MAE computed only on the peak-hour subset.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if peak_hours is None:
            # Fall back to top 25 % of demand as "peak"
            threshold = np.percentile(y_true, 75)
            mask = y_true >= threshold
        else:
            peak_hours = np.asarray(peak_hours)
            if peak_hours.dtype == bool:
                mask = peak_hours
            else:
                # Treat as integer indices
                mask = np.zeros(len(y_true), dtype=bool)
                mask[peak_hours] = True

        if not np.any(mask):
            return 0.0

        return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    @staticmethod
    def confidence_interval(
        y_pred: NDArray[np.floating],
        residuals: NDArray[np.floating],
        alpha: float = 0.95,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute prediction confidence intervals from historical residuals.

        Uses the empirical distribution of *residuals* (``y_true - y_pred``
        on a calibration set) to construct a symmetric interval around
        ``y_pred``.

        Parameters
        ----------
        y_pred:
            New predictions of shape ``(n,)``.
        residuals:
            Historical residuals from a calibration set.
        alpha:
            Confidence level, e.g. ``0.95`` for a 95 % interval.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lower_bound, upper_bound)`` arrays of shape ``(n,)``.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)

        tail = (1.0 - alpha) / 2.0
        lower_q = np.percentile(residuals, tail * 100.0)
        upper_q = np.percentile(residuals, (1.0 - tail) * 100.0)

        lower_bound = y_pred + lower_q
        upper_bound = y_pred + upper_q

        return lower_bound, upper_bound

    @classmethod
    def compute_all(
        cls,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
        peak_hours: NDArray[np.integer] | list[int] | None = None,
    ) -> dict[str, float]:
        """Compute every available metric in one call.

        Parameters
        ----------
        y_true:
            Ground-truth target values.
        y_pred:
            Predicted values.
        peak_hours:
            Optional peak-hour mask or indices forwarded to
            :meth:`peak_hour_error`.

        Returns
        -------
        dict[str, float]
            Mapping of metric name to scalar value.
        """
        return {
            "mae": cls.mae(y_true, y_pred),
            "rmse": cls.rmse(y_true, y_pred),
            "mape": cls.mape(y_true, y_pred),
            "r2": cls.r2(y_true, y_pred),
            "peak_hour_error": cls.peak_hour_error(y_true, y_pred, peak_hours),
        }
