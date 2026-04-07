# Troubleshooting

This document covers common issues encountered when setting up and running the Energy Demand Forecasting MLOps System, along with their solutions.

---

## Docker Issues

### Services fail to start

**Symptom:** `docker compose up` exits with errors or services keep restarting.

**Diagnosis:**

```bash
# Check status of all services
docker compose ps -a

# View logs for a specific failing service
docker compose logs --tail=50 <service-name>
```

**Common causes:**

1. **Docker daemon not running** -- Start Docker Desktop or run `sudo systemctl start docker`
2. **Insufficient memory** -- See the Memory Issues section below
3. **Stale containers from previous runs** -- Clean up with:
   ```bash
   docker compose down -v
   docker system prune -f
   docker compose up -d --build
   ```

### Docker Compose version mismatch

**Symptom:** `docker compose` command not found, or `version` key warnings.

**Solution:** This project uses Docker Compose V2 (the `docker compose` plugin, not the standalone `docker-compose` binary).

```bash
# Check your version
docker compose version

# If not installed, install the plugin
# Ubuntu/Debian:
sudo apt-get install docker-compose-plugin

# macOS/Windows: Update Docker Desktop to the latest version
```

### Build fails with network errors

**Symptom:** `pip install` inside Docker build fails to reach PyPI.

**Solution:**

```bash
# Check Docker DNS configuration
docker run --rm alpine nslookup pypi.org

# If DNS fails, add DNS to Docker daemon config (/etc/docker/daemon.json):
# { "dns": ["8.8.8.8", "8.8.4.4"] }
# Then restart Docker: sudo systemctl restart docker
```

### Container runs out of disk space

**Symptom:** Build fails with "no space left on device."

```bash
# Check Docker disk usage
docker system df

# Remove unused images, containers, and volumes
docker system prune -a --volumes
```

---

## MLflow Connection Errors

### MLflow UI not accessible

**Symptom:** http://localhost:5000 returns connection refused.

**Diagnosis:**

```bash
# Check if MLflow container is running and healthy
docker compose ps mlflow

# View MLflow logs
docker compose logs mlflow
```

**Common causes:**

1. **MLflow waiting for PostgreSQL** -- The MLflow container depends on PostgreSQL being healthy. Wait 30-60 seconds after `docker compose up`.
2. **PostgreSQL credentials mismatch** -- Verify that `POSTGRES_USER` and `POSTGRES_PASSWORD` in `.env` match what PostgreSQL was initialized with. If you changed these after first run, you need to remove the volume:
   ```bash
   docker compose down -v
   docker compose up -d
   ```

### MLflow cannot write artifacts

**Symptom:** Training runs fail with S3/MinIO connection errors.

**Diagnosis:**

```bash
# Check if MinIO is healthy
docker compose ps minio

# Check if the mlflow bucket exists
docker compose logs create-bucket
```

**Solution:**

```bash
# Manually create the bucket
docker compose run --rm create-bucket

# Or via MinIO client
docker run --rm --network energy-net minio/mc \
  sh -c "mc alias set myminio http://minio:9000 minioadmin minioadmin && mc mb --ignore-existing myminio/mlflow"
```

### "No model is currently loaded" when serving

**Symptom:** The API returns 503 with "No model is currently loaded."

**Cause:** No model has been trained and registered in MLflow's Model Registry with the Production stage.

**Solution:**

```bash
# Generate data and train a model
./scripts/seed_data.sh
./scripts/run_initial_training.sh

# Restart the API to pick up the new model
docker compose restart api
```

---

## Airflow DAG Import Errors

### DAGs not appearing in the Airflow UI

**Symptom:** The Airflow web UI shows no DAGs or a DAG import error.

**Diagnosis:**

```bash
# Check for import errors
docker compose exec airflow-scheduler airflow dags list-import-errors

# Verify DAG files are mounted correctly
docker compose exec airflow-scheduler ls -la /opt/airflow/dags/
```

**Common causes:**

1. **Python import error in DAG file** -- The DAG file imports modules that are not installed in the Airflow container. Check that `src/` is mounted and the package is installed.
2. **Syntax error** -- Any Python syntax error prevents the DAG from loading. Check the import error list.
3. **Volume mount issue** -- Verify that `./airflow/dags` is correctly mounted in `docker-compose.yml`.

### DAG tasks fail with ModuleNotFoundError

**Symptom:** Tasks fail with `ModuleNotFoundError: No module named 'energy_forecast'`.

**Solution:** The Airflow containers mount `./src` to `/opt/airflow/src`. Ensure the Airflow Dockerfile installs the package:

```bash
# Check if the package is installed
docker compose exec airflow-scheduler pip list | grep energy-forecast

# If not, rebuild the Airflow image
docker compose build airflow-webserver airflow-scheduler
docker compose up -d airflow-webserver airflow-scheduler
```

### Airflow database migration issues

**Symptom:** `airflow-init` fails with database errors.

```bash
# Reset the Airflow database (WARNING: loses task history)
docker compose exec airflow-scheduler airflow db reset -y
docker compose restart airflow-init
```

---

## GPU and CUDA Setup

### PyTorch does not detect GPU

**Symptom:** `torch.cuda.is_available()` returns `False`.

**Diagnosis:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**Common causes and solutions:**

1. **NVIDIA drivers not installed**
   ```bash
   # Check driver
   nvidia-smi

   # Install on Ubuntu
   sudo apt-get install nvidia-driver-535
   ```

2. **CUDA toolkit version mismatch** -- PyTorch requires a compatible CUDA version. Check the [PyTorch compatibility matrix](https://pytorch.org/get-started/locally/).

3. **Docker GPU access not configured**
   ```bash
   # Install NVIDIA Container Toolkit
   sudo apt-get install nvidia-container-toolkit
   sudo systemctl restart docker

   # Run container with GPU access
   docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

### LSTM training is very slow on CPU

The LSTM model is designed for GPU training. On CPU, a full training run can take hours. Options:

1. **Use XGBoost instead** -- It is the primary model and runs efficiently on CPU
2. **Reduce LSTM configuration** -- In `configs/model_config.yaml`:
   ```yaml
   lstm:
     hidden_size: 32      # was 128
     num_layers: 1         # was 2
     epochs: 10            # was 50
     sequence_length: 24   # was 168
   ```
3. **Use a cloud GPU** -- Google Colab, AWS SageMaker, or similar

---

## Port Conflicts

### Port already in use

**Symptom:** `Bind for 0.0.0.0:XXXX failed: port is already allocated`

**Diagnosis:**

```bash
# Find what is using the port (Linux/macOS)
lsof -i :5000
# or
sudo ss -tlnp | grep 5000
```

**Solution options:**

1. **Stop the conflicting process:**
   ```bash
   kill <PID>
   ```

2. **Change the port in docker-compose.yml.** For example, to move MLflow from 5000 to 5001:
   ```yaml
   mlflow:
     ports:
       - "5001:5000"
   ```
   Then update `MLFLOW_TRACKING_URI` in `.env` to `http://localhost:5001`.

### Default port assignments

| Port | Service | Common Conflicts |
|---|---|---|
| 3000 | Grafana | Other Grafana instances, some Node.js apps |
| 5000 | MLflow | macOS AirPlay Receiver (disable in System Settings) |
| 5432 | PostgreSQL | Local PostgreSQL installation |
| 8000 | FastAPI | Other web servers |
| 8080 | Airflow | Jenkins, other Java apps |
| 9000 | MinIO (API) | SonarQube |
| 9001 | MinIO (Console) | -- |
| 9090 | Prometheus | -- |

### macOS port 5000 conflict

On macOS Monterey and later, AirPlay Receiver uses port 5000 by default.

**Solution:** Disable AirPlay Receiver in System Settings > General > AirDrop and Handoff > AirPlay Receiver, or change the MLflow port as described above.

---

## Memory Issues

### Docker containers keep getting killed (OOMKilled)

**Symptom:** Containers restart with exit code 137 or show `OOMKilled: true` in `docker inspect`.

**Solution:**

1. **Increase Docker memory allocation:**
   - **macOS / Windows:** Docker Desktop > Settings > Resources > Memory > set to 8 GB or more
   - **Linux:** Docker uses available system RAM directly; check with `free -h`

2. **Reduce the number of running services.** If you are only working on specific modules, start only what you need:
   ```bash
   # Data and training work (no monitoring stack)
   docker compose up -d postgres minio mlflow

   # API development (no Airflow)
   docker compose up -d postgres minio mlflow api prometheus grafana
   ```

3. **Limit container memory in docker-compose.yml:**
   ```yaml
   services:
     mlflow:
       deploy:
         resources:
           limits:
             memory: 1G
   ```

### Python process runs out of memory during training

**Symptom:** `MemoryError` or the process is killed during feature engineering or model training.

**Solutions:**

1. **Reduce data size** -- Generate fewer buildings or a shorter date range:
   ```python
   gen = SyntheticDataGenerator(num_buildings=10, start_date='2022-01-01', end_date='2023-12-31')
   ```

2. **Use chunked processing** -- Process data in chunks instead of loading everything into memory.

3. **Reduce XGBoost memory** -- Lower `n_estimators` or `max_depth` in `configs/model_config.yaml`.

4. **Use float32 instead of float64:**
   ```python
   df = df.astype({col: 'float32' for col in df.select_dtypes('float64').columns})
   ```

---

## Network Issues

### Services cannot communicate with each other

**Symptom:** One service cannot reach another (e.g., API cannot connect to MLflow).

**Diagnosis:**

```bash
# Check that all services are on the same network
docker network inspect energy-demand-forecasting-mlops-system_energy-net

# Test connectivity from one container to another
docker compose exec api curl -s http://mlflow:5000/health
```

**Solution:** Ensure all services are on the `energy-net` network in `docker-compose.yml`. If you added a new service, include:

```yaml
networks:
  - energy-net
```

### DNS resolution fails inside containers

**Symptom:** Services fail to resolve hostnames like `postgres` or `mlflow`.

```bash
# Restart the Docker network
docker compose down
docker compose up -d
```

---

## Python Environment Issues

### Conflicting package versions

**Symptom:** `ImportError` or `AttributeError` after installing.

```bash
# Create a clean virtual environment
python -m venv .venv --clear
source .venv/bin/activate
pip install -e ".[dev,test]"
```

### Wrong Python version

**Symptom:** `SyntaxError` from type hints or f-strings.

```bash
# Verify Python version (must be 3.10+)
python --version

# Use pyenv to install the correct version
pyenv install 3.11.7
pyenv local 3.11.7
```

---

## Getting More Help

If your issue is not listed here:

1. **Check the logs** -- Almost every problem leaves a trace in Docker logs (`docker compose logs <service>`)
2. **Search existing issues** -- Check the GitHub Issues for similar problems
3. **Reproduce minimally** -- Isolate the failing component and test it independently
4. **Open an issue** -- Include: the exact error message, relevant logs, your OS, Docker version, and Python version
