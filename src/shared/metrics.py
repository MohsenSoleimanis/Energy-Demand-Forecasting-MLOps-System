"""ML metrics computation shared across training, evaluation, and monitoring.

All functions are pure: no side effects, no I/O.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Mean Absolute Percentage Error.

    Observations where *y_true* equals zero are excluded to avoid
    division-by-zero.  Returns ``float('nan')`` when no valid observations
    remain or when inputs are empty.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a fraction (e.g. 0.05 means 5 %).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Compute standard regression metrics.

    Returns a dict with keys ``mae``, ``rmse``, ``mape``, and ``r2``.
    Handles empty arrays gracefully (all values become ``nan``).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric name to value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = compute_mape(y_true, y_pred)
    r2 = _compute_r2(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination).

    Returns ``nan`` when variance of *y_true* is zero.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
