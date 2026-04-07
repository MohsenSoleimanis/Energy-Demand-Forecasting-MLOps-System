# Energy Demand Forecasting MLOps System

[![CI](https://github.com/your-org/Energy-Demand-Forecasting-MLOps-System/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/Energy-Demand-Forecasting-MLOps-System/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-grade MLOps platform for building-level energy demand forecasting. This system covers the complete machine learning lifecycle -- from synthetic data generation and experiment tracking to model serving, monitoring, and orchestrated retraining -- all running locally with Docker Compose.

Built as a hands-on workshop for learning MLOps best practices with real-world patterns.

---

## What You Will Learn

- **Data Engineering** -- Synthetic data generation with realistic seasonality, data validation with Great Expectations, feature engineering pipelines
- **Experiment Tracking** -- MLflow for parameter logging, metric comparison, and artifact storage with MinIO
- **Model Training** -- Ridge regression baselines, XGBoost gradient boosting, LSTM deep learning, Optuna hyperparameter tuning, time-series cross-validation
- **Model Serving** -- FastAPI REST API with Pydantic validation, batch inference, confidence intervals
- **Monitoring** -- Prometheus metrics, Grafana dashboards, Evidently drift detection, automated alerting
- **CI/CD** -- GitHub Actions pipelines, Docker builds, Kubernetes deployment manifests
- **Orchestration** -- Apache Airflow DAGs for scheduled retraining with dependency management

---

## Architecture Overview

```
                    +-------------------+
                    |   Apache Airflow  |
                    |   (Orchestration) |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+       +----------v---------+
    |  Data Pipeline      |       |  Training Pipeline  |
    |  - Generation       |       |  - XGBoost / LSTM   |
    |  - Validation       |       |  - Cross-validation  |
    |  - Feature Eng.     |       |  - Hyperparameter    |
    +---------+----------+       +----------+----------+
              |                             |
              +-------------+---------------+
                            |
                   +--------v--------+       +-----------------+
                   |     MLflow      |<----->|      MinIO      |
                   | (Tracking +     |       | (S3 Artifacts)  |
                   |  Registry)      |       +-----------------+
                   +--------+--------+
                            |
                   +--------v--------+
                   |    FastAPI      |
                   | (Prediction API)|
                   +--------+--------+
                            |
              +-------------+---------------+
              |                             |
    +---------v----------+       +----------v----------+
    |    Prometheus       |       |      Grafana        |
    |  (Metrics Store)    |       |   (Dashboards)      |
    +--------------------+       +---------------------+
```

For detailed architecture diagrams with Mermaid, see [docs/architecture.md](docs/architecture.md).

---

## Quick Start

Get the entire platform running in three steps:

### 1. Set up infrastructure

```bash
git clone https://github.com/your-org/Energy-Demand-Forecasting-MLOps-System.git
cd Energy-Demand-Forecasting-MLOps-System
./scripts/setup_infrastructure.sh
```

### 2. Generate data

```bash
./scripts/seed_data.sh
```

### 3. Train your first model

```bash
./scripts/run_initial_training.sh
```

That is it. Open http://localhost:5000 to see your experiment in MLflow, and http://localhost:8000/docs to test the prediction API.

---

## Project Structure

```
Energy-Demand-Forecasting-MLOps-System/
|
|-- src/energy_forecast/            # Core Python package
|   |-- data/                       # Data generation, validation, processing
|   |   |-- synthetic.py            #   Realistic synthetic data generator
|   |   |-- validator.py            #   Great Expectations validation
|   |   |-- processor.py            #   Cleaning and imputation
|   |   +-- schemas.py              #   Data schemas
|   |-- features/                   # Feature engineering
|   |   |-- engineering.py          #   Temporal, lag, rolling features
|   |   +-- store.py                #   Feature store interface
|   |-- models/                     # Model implementations
|   |   |-- base.py                 #   Abstract base forecaster
|   |   |-- linear.py               #   Ridge regression baseline
|   |   |-- xgboost_model.py        #   XGBoost gradient boosting
|   |   |-- lstm_model.py           #   PyTorch LSTM network
|   |   +-- registry.py             #   MLflow model registry client
|   |-- training/                   # Training infrastructure
|   |   |-- trainer.py              #   Training loop with MLflow logging
|   |   |-- cross_validation.py     #   Time-series CV strategies
|   |   +-- hyperparameter.py       #   Optuna integration
|   |-- evaluation/                 # Model evaluation
|   |   |-- metrics.py              #   RMSE, MAE, MAPE, R-squared
|   |   +-- evaluator.py            #   Evaluation orchestrator
|   |-- serving/                    # FastAPI prediction service
|   |   |-- app.py                  #   Application factory
|   |   |-- routes.py               #   API endpoints
|   |   |-- schemas.py              #   Request/response models
|   |   |-- dependencies.py         #   Model manager (DI)
|   |   +-- middleware.py           #   Request logging
|   |-- monitoring/                 # Production monitoring
|   |   |-- drift.py                #   Evidently drift detection
|   |   |-- prometheus_metrics.py   #   Prometheus metric definitions
|   |   +-- performance.py          #   Accuracy tracking
|   |-- pipelines/                  # End-to-end pipelines
|   |   |-- training_pipeline.py    #   Full training workflow
|   |   +-- inference_pipeline.py   #   Batch inference workflow
|   +-- utils/                      # Shared utilities
|       |-- config.py               #   YAML config loader
|       |-- io.py                   #   File I/O helpers
|       +-- logging.py              #   Structured logging setup
|
|-- configs/                        # Configuration files
|   |-- data_config.yaml            #   Data generation parameters
|   |-- model_config.yaml           #   Model hyperparameters
|   |-- serving_config.yaml         #   API settings
|   |-- monitoring_config.yaml      #   Drift thresholds
|   +-- logging_config.yaml         #   Logging configuration
|
|-- airflow/                        # Airflow orchestration
|   |-- dags/
|   |   +-- training_dag.py         #   Weekly retraining DAG
|   +-- Dockerfile                  #   Airflow container image
|
|-- monitoring/                     # Monitoring configuration
|   |-- prometheus/
|   |   +-- prometheus.yml          #   Scrape targets
|   +-- grafana/
|       |-- provisioning/           #   Auto-provisioned datasources
|       +-- dashboards/             #   Pre-built dashboards
|
|-- deployment/                     # Deployment manifests
|   +-- kubernetes/
|       |-- namespace.yaml
|       |-- configmap.yaml
|       |-- deployment.yaml
|       |-- service.yaml
|       +-- hpa.yaml                #   Horizontal Pod Autoscaler
|
|-- .github/workflows/             # CI/CD pipelines
|   |-- ci.yml                      #   Lint, type-check, test
|   |-- deploy.yml                  #   Build and deploy
|   +-- integration-tests.yml       #   End-to-end tests
|
|-- scripts/                        # Automation scripts
|   |-- setup_infrastructure.sh     #   One-command platform setup
|   |-- seed_data.sh                #   Generate synthetic data
|   +-- run_initial_training.sh     #   First model training
|
|-- tests/                          # Test suites
|   |-- unit/                       #   Fast isolated tests
|   |-- integration/                #   Cross-component tests
|   +-- performance/                #   Benchmark tests
|
|-- notebooks/                      # Jupyter notebooks for exploration
|-- docs/                           # Documentation
|-- docker-compose.yml              # Full local stack definition
|-- Dockerfile                      # API service image
|-- Makefile                        # Developer task shortcuts
|-- pyproject.toml                  # Python project metadata
+-- .env.example                    # Environment variable template
```

---

## Workshop Modules

| Module | Topic | Duration | Key Tools |
|---|---|---|---|
| 1 | Data Engineering | 1.5-2h | Pandas, Great Expectations, Parquet |
| 2 | Experiment Tracking | 1.5-2h | MLflow, MinIO, PostgreSQL |
| 3 | Model Training | 2-3h | XGBoost, PyTorch LSTM, Optuna |
| 4 | Model Serving | 1.5-2h | FastAPI, Pydantic, REST APIs |
| 5 | Monitoring & Observability | 2-2.5h | Prometheus, Grafana, Evidently |
| 6 | CI/CD & Deployment | 1.5-2h | GitHub Actions, Docker, Kubernetes |
| 7 | Orchestration | 1.5-2h | Apache Airflow, DAGs |

See [docs/workshop_modules.md](docs/workshop_modules.md) for full module details with exercises and expected outcomes.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Language** | Python | 3.10+ |
| **ML Models** | XGBoost, PyTorch, scikit-learn | 2.0+, 2.1+, 1.3+ |
| **Hyperparameter Tuning** | Optuna | 3.4+ |
| **Experiment Tracking** | MLflow | 2.9+ |
| **API Framework** | FastAPI + Uvicorn | 0.104+ |
| **Data Validation** | Great Expectations | 0.18+ |
| **Drift Detection** | Evidently | 0.4+ |
| **Orchestration** | Apache Airflow | 2.8+ |
| **Metrics** | Prometheus | 2.48 |
| **Dashboards** | Grafana | 10.2 |
| **Artifact Storage** | MinIO (S3-compatible) | latest |
| **Database** | PostgreSQL | 15 |
| **Containerization** | Docker Compose | V2 |
| **CI/CD** | GitHub Actions | -- |
| **Deployment** | Kubernetes | -- |
| **Linting** | Ruff | 0.1.8+ |
| **Type Checking** | mypy | 1.7+ |

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| Docker | 24.0+ |
| Docker Compose | V2 (2.20+) |
| RAM | 8 GB (16 GB recommended) |
| Disk | 10 GB free |
| GPU | Optional (for LSTM training) |

See [docs/setup.md](docs/setup.md) for detailed installation instructions.

---

## Common Commands

```bash
# Start everything
make docker-up

# Stop everything
make docker-down

# Generate synthetic data
make seed

# Train a model
make train

# Start API in dev mode (with auto-reload)
make serve

# Run tests
make test

# Lint and format
make lint
make format

# Type checking
make type-check
```

---

## Services (Docker Compose)

| Service | URL | Credentials |
|---|---|---|
| MLflow UI | http://localhost:5000 | -- |
| FastAPI Docs | http://localhost:8000/docs | -- |
| Airflow UI | http://localhost:8080 | admin / admin |
| Prometheus | http://localhost:9090 | -- |
| Grafana | http://localhost:3000 | admin / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, data flows, and technology rationale |
| [Setup Guide](docs/setup.md) | Installation, configuration, and verification |
| [API Reference](docs/api_reference.md) | Endpoints, request/response schemas, error codes |
| [Workshop Modules](docs/workshop_modules.md) | 7-module curriculum with exercises |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository and create a feature branch from `main`
2. **Install dev dependencies:** `pip install -e ".[dev,test]"`
3. **Write tests** for new functionality
4. **Run the full check suite** before submitting:
   ```bash
   make lint && make type-check && make test
   ```
5. **Submit a pull request** with a clear description of the changes

### Code Style

- This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length limit: 100 characters
- Type hints required for all function signatures (enforced by mypy)
- Follow existing patterns in the codebase

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
