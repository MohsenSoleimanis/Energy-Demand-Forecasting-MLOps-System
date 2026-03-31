"""ML training layer for Belgian energy demand forecasting.

Modules:
    data       -- Data loading and temporal splitting.
    train      -- LightGBM model building, training, and Optuna tuning.
    evaluate   -- Model evaluation, sliced metrics, and diagnostic plots.
    register   -- Quality-gate checks and MLflow Model Registry integration.
    baseline   -- Persistence and linear-regression baseline models.
"""
