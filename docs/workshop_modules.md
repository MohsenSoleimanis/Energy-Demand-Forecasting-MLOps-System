# Workshop Modules

A hands-on curriculum for learning production MLOps through Energy Demand Forecasting.

---

## Module 1: Data Engineering

**Learning Objectives:**
- Design data pipelines for ML systems
- Generate realistic synthetic data
- Implement data validation
- Build a feature store

**Key Files:** `src/energy_forecast/data/`, `src/energy_forecast/features/`

**Exercises:**

1. Generate synthetic data with `SyntheticDataGenerator`
2. Validate data quality with `DataValidator`
3. Engineer features (lag, rolling, cyclical time, weather)
4. Save versioned features to the `FeatureStore`

**Expected Outcome:** 50+ engineered features validated and stored

---

## Module 2: Experiment Tracking with MLflow

**Learning Objectives:**
- Set up MLflow for experiment tracking
- Log parameters, metrics, and artifacts
- Compare model runs
- Use the model registry for versioning

**Key Files:** `src/energy_forecast/models/registry.py`, `src/energy_forecast/training/trainer.py`

**Exercises:**

1. Start MLflow server (`docker compose up -d mlflow`)
2. Train models and log runs with `ModelRegistry.log_run()`
3. Compare experiments with `ModelRegistry.compare_runs()`
4. Register and promote models through stages

**Expected Outcome:** Models tracked in MLflow UI with metrics comparison

---

## Module 3: Model Training

**Learning Objectives:**
- Implement multiple model architectures
- Use time-series cross-validation
- Perform hyperparameter optimization with Optuna

**Key Files:** `src/energy_forecast/models/`, `src/energy_forecast/training/`

**Exercises:**

1. Train Linear baseline, XGBoost, and LSTM models
2. Run time-series cross-validation with `TimeSeriesCV`
3. Optimize hyperparameters with `HyperparameterSearcher`
4. Compare model performance

**Expected Outcome:** Three trained models, optimized hyperparameters, comparison table

---

## Module 4: Model Serving

**Learning Objectives:**
- Build REST APIs for ML inference
- Implement input validation and error handling
- Add request logging and metrics

**Key Files:** `src/energy_forecast/serving/`

**Exercises:**

1. Start the API (`make serve`)
2. Make single predictions via `POST /predict`
3. Make batch predictions via `POST /predict/batch`
4. Check health via `GET /health`
5. Review Swagger docs at `/docs`

**Expected Outcome:** Running API with predictions and confidence intervals

---

## Module 5: Monitoring & Observability

**Learning Objectives:**
- Detect data drift using statistical tests
- Track model performance over time
- Build monitoring dashboards

**Key Files:** `src/energy_forecast/monitoring/`, `monitoring/`

**Exercises:**

1. Detect drift with `DriftDetector` using reference vs current data
2. Simulate drift with `SyntheticDataGenerator.generate_with_drift()`
3. Track performance with `PerformanceMonitor`
4. View Grafana dashboards at http://localhost:3000

**Expected Outcome:** Drift detection reports, performance history, live dashboards

---

## Module 6: CI/CD & Deployment

**Learning Objectives:**
- Set up automated CI pipelines
- Build and deploy Docker images
- Deploy to Kubernetes

**Key Files:** `.github/workflows/`, `Dockerfile`, `deployment/`

**Exercises:**

1. Review CI pipeline (lint -> type-check -> test)
2. Build Docker image (`docker build -t energy-forecast .`)
3. Review Kubernetes manifests
4. Review Terraform infrastructure as code

**Expected Outcome:** CI pipeline, Docker image, K8s deployment config

---

## Module 7: Orchestration with Airflow

**Learning Objectives:**
- Design DAGs for ML workflows
- Schedule training, inference, and monitoring

**Key Files:** `airflow/dags/`

**Exercises:**

1. Start Airflow (`docker compose up -d airflow-webserver`)
2. Review training, inference, and monitoring DAGs
3. Trigger manual DAG runs
4. Monitor task execution

**Expected Outcome:** Three scheduled DAGs executing ML pipelines
