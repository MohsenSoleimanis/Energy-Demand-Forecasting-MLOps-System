# Belgian Energy Demand Forecasting MLOps System

Production-grade MLOps system for predicting hourly electricity demand (MW) for the Belgian grid zone 24 hours ahead.

## Overview

- **Model**: LightGBM (primary) with persistence and linear regression baselines
- **Data Sources**: ENTSO-E Transparency Platform (load, price, generation), Open-Meteo (weather), Belgian calendar
- **Architecture**: Medallion (Bronze/Silver/Gold) with dbt + DuckDB
- **ML Tracking**: MLflow with model registry
- **Serving**: FastAPI with Prometheus metrics
- **Monitoring**: Evidently for drift detection, Grafana dashboards
- **Orchestration**: Airflow DAGs for data refresh, batch forecasting, retraining, monitoring
- **Target**: MAPE < 5% on test set

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- ENTSO-E API key (free, email transparency@entsoe.eu)

### Setup

```bash
# Clone and configure
cp .env.example .env
# Edit .env to add your ENTSOE_API_KEY

# Start infrastructure
make up

# Install Python dependencies
pip install -e ".[dev]"

# Run data ingestion
make ingest

# Run dbt transformations (bronze -> silver -> gold)
make transform

# Train model
make train

# Start serving API
make serve

# Run tests
make test
```

### Services

| Service | URL | Purpose |
|---------|-----|---------|
| MLflow | http://localhost:5000 | Experiment tracking & model registry |
| MinIO Console | http://localhost:9001 | Object storage management |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| FastAPI | http://localhost:8000 | Prediction API |

## Architecture

```
ENTSO-E API ──┐
Open-Meteo ───┤──> Bronze (Parquet/MinIO) ──> Silver (dbt views) ──> Gold (dbt tables)
Calendar ─────┘                                                          │
                                                                         ├──> Training (LightGBM + MLflow)
                                                                         ├──> Serving (FastAPI)
                                                                         └──> Monitoring (Evidently + Grafana)
```

### Data Pipeline
1. **Ingestion**: Python scripts pull from ENTSO-E and Open-Meteo APIs into Parquet files (Bronze)
2. **Transformation**: dbt models clean, deduplicate, join, and compute features (Silver to Gold)
3. **Validation**: Great Expectations suites validate data quality at each layer

### ML Pipeline (DVC)
1. **Pull Gold**: Export training data from DuckDB
2. **Train**: LightGBM with temporal cross-validation
3. **Evaluate**: Comprehensive metrics + SHAP explanations
4. **Register**: Quality-gated model registration in MLflow

### Serving
- FastAPI with `/predict`, `/predict/batch`, `/health`, `/metrics` endpoints
- Hot-reload model from MLflow registry
- Prometheus instrumentation for latency, throughput, prediction distribution

## Project Structure

```
├── src/
│   ├── data_platform/          # Data engineering
│   │   ├── ingestion/          # ENTSO-E, weather, calendar scripts
│   │   ├── dbt_project/        # dbt models (staging + marts)
│   │   └── quality/            # Great Expectations suites
│   └── ml/                     # ML engineering
│       ├── features/           # Shared feature engineering module
│       ├── training/           # Train, evaluate, register, baselines
│       ├── serving/            # FastAPI app, schemas, metrics
│       └── monitoring/         # Drift detection, performance reports, alerting
├── configs/                    # Hydra YAML configurations
├── tests/                      # Unit + integration tests
├── airflow/dags/               # Orchestration DAGs
├── docker/                     # Training + serving Dockerfiles
├── grafana/dashboards/         # Pre-configured Grafana dashboards
└── prometheus/                 # Prometheus scrape config
```

## Key Design Decisions

1. **Point-in-time correctness**: Features use only past data (LAG/shift), target uses future data (LEAD). Training uses weather actuals; serving uses weather forecasts.
2. **No training-serving skew**: Same `prepare_features()` function used in both paths.
3. **Temporal splits**: No random splits. Train < val_end < test, sorted by timestamp.
4. **DuckDB for analytics**: Lightweight, embedded OLAP engine - no separate database server needed for dbt.
5. **MinIO for storage**: S3-compatible object storage for Parquet files, MLflow artifacts, and DVC remote.

## License

MIT
