# Setup Guide

This guide walks you through setting up the Energy Demand Forecasting MLOps System for local development and experimentation.

---

## Prerequisites

### Required Software

| Software | Minimum Version | Check Command |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Docker | 24.0+ | `docker --version` |
| Docker Compose | 2.20+ (V2 plugin) | `docker compose version` |
| Git | 2.30+ | `git --version` |
| Make | 4.0+ | `make --version` |

### Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB free |
| CPU | 4 cores | 8 cores |
| GPU | Not required | NVIDIA GPU with CUDA 11.8+ (for LSTM training) |

Docker needs at least 6 GB of memory allocated. On macOS/Windows, check Docker Desktop settings under Resources.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/Energy-Demand-Forecasting-MLOps-System.git
cd Energy-Demand-Forecasting-MLOps-System
```

### 2. One-Command Setup (Recommended)

The setup script creates the environment file, builds all Docker services, and starts the platform:

```bash
chmod +x scripts/setup_infrastructure.sh
./scripts/setup_infrastructure.sh
```

This will:
- Copy `.env.example` to `.env` (if `.env` does not exist)
- Create `data/raw`, `data/processed`, `data/features`, and `logs` directories
- Build and start all Docker Compose services
- Print URLs and default credentials for every service

### 3. Generate Synthetic Data

```bash
chmod +x scripts/seed_data.sh
./scripts/seed_data.sh
```

This generates approximately 1.75 million rows of hourly energy data for 50 buildings spanning 2020 through 2023.

### 4. Run Initial Training

```bash
chmod +x scripts/run_initial_training.sh
./scripts/run_initial_training.sh
```

This trains an XGBoost model, logs the experiment to MLflow, and registers the model for serving.

---

## Environment Configuration

The `.env` file controls all service configuration. Copy the example and customize as needed:

```bash
cp .env.example .env
```

### Key Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `POSTGRES_USER` | `mlflow` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `mlflow_password` | PostgreSQL password |
| `POSTGRES_DB` | `mlflow` | PostgreSQL database name |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO secret key |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |
| `API_MODEL_STAGE` | `Production` | Which MLflow model stage to load |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana admin password |

### Configuration Files

Additional YAML configuration files live in the `configs/` directory:

| File | Purpose |
|---|---|
| `data_config.yaml` | Data generation and processing parameters |
| `model_config.yaml` | Model architectures and hyperparameters |
| `serving_config.yaml` | API and serving settings |
| `monitoring_config.yaml` | Drift thresholds and alert rules |
| `logging_config.yaml` | Structured logging configuration |

---

## Running with Docker (Full Stack)

### Start All Services

```bash
docker compose up -d --build
```

Or use Make:

```bash
make docker-up
```

### Check Service Health

```bash
docker compose ps
```

All services should show `healthy` or `running` status. Give Airflow about 60 seconds for the webserver health check to pass.

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f mlflow
```

### Stop All Services

```bash
docker compose down
```

To also remove persistent volumes (databases, artifacts):

```bash
make docker-down
```

### Service URLs

| Service | URL | Default Credentials |
|---|---|---|
| MLflow UI | http://localhost:5000 | None |
| FastAPI Docs | http://localhost:8000/docs | None |
| Airflow UI | http://localhost:8080 | admin / admin |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

---

## Running Locally (Development Mode)

For faster iteration, you can run Python components directly while keeping infrastructure services in Docker.

### 1. Start Infrastructure Only

```bash
docker compose up -d postgres minio mlflow prometheus grafana
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install the Package

```bash
# Production dependencies
pip install -e .

# Development dependencies (linting, testing, notebooks)
pip install -e ".[dev,test]"
```

### 4. Generate Data and Train

```bash
make seed
make train
```

### 5. Start the API Server

```bash
make serve
```

This runs uvicorn with auto-reload enabled for development.

### 6. Run Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage report
make test-cov
```

### 7. Lint and Format

```bash
make lint
make format
make type-check
```

---

## Verifying the Setup

After completing the setup, verify each component is working:

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1",
  "uptime_seconds": 42.0
}
```

### 2. Make a Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T14:00:00",
    "building_id": "BLDG_001",
    "temperature": 22.5,
    "humidity": 45.0
  }'
```

### 3. Check MLflow Experiments

Open http://localhost:5000 in your browser. You should see the `energy-demand-forecasting` experiment with at least one completed run.

### 4. Check Airflow DAGs

Open http://localhost:8080 and log in with admin/admin. You should see the `energy_demand_training` DAG listed.

### 5. Check Grafana Dashboards

Open http://localhost:3000, log in with admin/admin, and navigate to the pre-provisioned dashboards.

### 6. Check Prometheus Targets

Open http://localhost:9090/targets. The API target should show as UP.

---

## Troubleshooting Common Issues

### Docker Services Fail to Start

**Symptom:** `docker compose up` exits with errors.

```bash
# Check which services failed
docker compose ps -a

# View logs for the failing service
docker compose logs <service-name>

# Restart a specific service
docker compose restart <service-name>
```

### Port Already in Use

**Symptom:** `Bind for 0.0.0.0:5000 failed: port is already allocated`

```bash
# Find what is using the port
lsof -i :5000

# Either stop the conflicting process or change the port in .env / docker-compose.yml
```

### MLflow Cannot Connect to PostgreSQL

**Symptom:** MLflow logs show `OperationalError: could not connect to server`

Wait 30 seconds for PostgreSQL to finish initializing. If it persists:

```bash
docker compose restart mlflow
```

### Python Package Installation Fails

**Symptom:** `pip install -e .` fails with build errors.

```bash
# Ensure Python 3.10+
python --version

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python3-dev build-essential
```

### Insufficient Docker Memory

**Symptom:** Services crash with OOM errors or Docker becomes unresponsive.

Increase Docker memory allocation to at least 6 GB:
- **macOS / Windows:** Docker Desktop > Settings > Resources > Memory
- **Linux:** Check available RAM with `free -h`

See [docs/troubleshooting.md](troubleshooting.md) for a complete list of known issues and solutions.
