# ADR-001: Full Codebase Rewrite — Engineering Principles & Architecture

**Status:** Accepted
**Date:** 2026-03-31
**Author:** Mohsen Soleimani

## Context

The initial codebase was generated rapidly and accumulated technical debt:
- SRP violations (130+ line functions doing 7 things)
- SSoT violations (config in code AND yaml, both different)
- DRY violations (upload_parquet_to_s3 duplicated 4x, metrics computation 4x)
- Hardcoded magic numbers, paths, credentials
- Silent error swallowing instead of fail-fast
- Missing type hints on all public functions
- Global mutable state in serving layer

## Decision

Complete rewrite following these principles:

### 1. Single Responsibility Principle (SRP)
- Every function does ONE thing
- Max ~30 lines per function
- train() becomes: load_data() → split_data() → build_model() → evaluate_model() → log_artifacts()

### 2. Single Source of Truth (SSoT)
- ONE config file per concern (thresholds.yaml, features.yaml, etc.)
- Code READS config, never defines its own defaults that conflict
- If config says train_ratio=0.70, the code uses exactly that

### 3. DRY (Don't Repeat Yourself)
- src/shared/s3.py: ONE upload_parquet function used everywhere
- src/shared/metrics.py: ONE compute_metrics function
- src/shared/config.py: ONE config loader

### 4. Robustness & Environment Agnosticism
- ZERO hardcoded paths — all from config or env vars
- ZERO magic numbers — all named constants from config
- Works on Windows, Linux, Docker identically

### 5. Fail-Fast Principle
- If --tune is requested but Optuna not installed → RAISE, don't fallback
- If config file missing → RAISE, don't use silent defaults
- If S3 upload fails → RAISE, don't swallow
- Use specific exceptions, not bare except

### 6. Readability
- Type hints on ALL public functions
- Docstrings on ALL modules and public functions
- No inline classes (callbacks get their own file)
- Imports sorted (isort/ruff)

### 7. Dependency Injection
- No global mutable state (_model, _prediction_logger)
- Use FastAPI's dependency injection and lifespan state
- Services receive their dependencies, don't create them

## File Structure After Rewrite

```
src/
├── shared/                          # Cross-cutting concerns
│   ├── __init__.py
│   ├── config.py                    # Config loader (reads YAML + .env)
│   ├── s3.py                        # S3/MinIO operations (upload, list, read)
│   ├── metrics.py                   # ML metrics computation (MAE, RMSE, MAPE, R²)
│   ├── logging_config.py            # Structured JSON logging
│   └── exceptions.py                # Custom exceptions (ConfigError, DataError, etc.)
│
├── data_platform/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── entsoe.py                # ENTSO-E ingestion (load, price, generation)
│   │   ├── weather.py               # Open-Meteo historical weather
│   │   ├── weather_forecast.py      # Open-Meteo forecast snapshots
│   │   └── calendar.py              # Belgian calendar generation
│   ├── dbt_project/                 # Unchanged (SQL, YAML)
│   └── quality/                     # Great Expectations
│
├── ml/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py           # Feature computation (reads window sizes from config)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data.py                  # Data loading + splitting
│   │   ├── train.py                 # Model training only
│   │   ├── evaluate.py              # Evaluation + plots
│   │   ├── register.py              # Model registration + quality gates
│   │   └── baseline.py              # Baseline models
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI app (thin — delegates to services)
│   │   ├── schemas.py               # Pydantic models
│   │   ├── auth.py                  # API key authentication
│   │   ├── model_service.py         # Model loading, prediction, shadow
│   │   ├── prediction_logger.py     # Audit trail (buffered S3 writer)
│   │   └── rollback.py              # Auto-rollback on degradation
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py                 # Evidently drift reports
│       ├── performance.py           # Performance monitoring
│       ├── alerting.py              # Tiered alerts + event-driven retrain
│       └── feedback.py              # Ground truth feedback loop
│
configs/
├── data/
│   ├── ingestion.yaml               # API endpoints, coordinates, buckets
│   └── features.yaml                # Window sizes, feature list
├── training/
│   └── lightgbm.yaml                # Hyperparams, split ratios, quality gates
├── serving/
│   └── api.yaml                     # Model name, batch limits, auth config
└── monitoring/
    └── thresholds.yaml              # All thresholds (MAPE, PSI, latency) — SSoT
```

## Config SSoT Rules

- **thresholds.yaml** is the ONLY place MAPE/PSI thresholds live
- **features.yaml** is the ONLY place feature list + window sizes live
- **lightgbm.yaml** is the ONLY place split ratios + hyperparams live
- **ingestion.yaml** is the ONLY place coordinates + bucket names live
- Code NEVER defines fallback defaults for these values

## Error Handling Rules

```python
# WRONG - silent swallow
try:
    upload(data)
except Exception:
    logger.exception("Upload failed")

# RIGHT - fail fast
try:
    upload(data)
except boto3.exceptions.S3UploadFailedError as e:
    raise DataPipelineError(f"Failed to upload to {bucket}/{key}") from e
```

## Type Hint Rules

```python
# WRONG
def compute_metrics(y_true, y_pred):

# RIGHT
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
```
