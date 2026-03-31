"""Business metrics translating forecast errors to EUR costs.

Belgian imbalance market (Elia):
- Under-forecast (actual > predicted): buy at premium ~100 EUR/MWh
- Over-forecast (actual < predicted): sell at discount ~30 EUR/MWh

These asymmetric costs reflect that under-forecasting is more
dangerous for grid stability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_UNDER_COST_EUR_MWH = 100.0  # Premium for buying on imbalance
DEFAULT_OVER_COST_EUR_MWH = 30.0    # Discount for selling surplus


def compute_imbalance_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    under_cost: float = DEFAULT_UNDER_COST_EUR_MWH,
    over_cost: float = DEFAULT_OVER_COST_EUR_MWH,
) -> dict[str, float]:
    """Compute asymmetric imbalance cost in EUR.

    Positive error (over-forecast) means the grid sold surplus at a
    discount.  Negative error (under-forecast) means the grid had to
    buy at a premium.

    Args:
        y_true: Actual load values (MW).
        y_pred: Predicted load values (MW).
        under_cost: EUR/MWh cost when actual > predicted.
        over_cost: EUR/MWh cost when actual < predicted.

    Returns:
        Dictionary with ``total_imbalance_cost_eur``,
        ``avg_hourly_cost_eur``, ``under_forecast_cost_eur``,
        ``over_forecast_cost_eur``, ``n_under_forecast_hours``,
        and ``n_over_forecast_hours``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    error = y_pred - y_true  # positive = over-forecast, negative = under-forecast

    under_mask = error < 0   # under-forecasted (need to buy)
    over_mask = error > 0    # over-forecasted (need to sell)

    under_total = float(np.abs(error[under_mask]).sum() * under_cost)
    over_total = float(error[over_mask].sum() * over_cost)
    total = under_total + over_total

    return {
        "total_imbalance_cost_eur": total,
        "avg_hourly_cost_eur": total / max(len(y_true), 1),
        "under_forecast_cost_eur": under_total,
        "over_forecast_cost_eur": over_total,
        "n_under_forecast_hours": int(under_mask.sum()),
        "n_over_forecast_hours": int(over_mask.sum()),
    }
