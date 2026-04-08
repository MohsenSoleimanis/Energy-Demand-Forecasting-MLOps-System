# Workshop Modules

This curriculum guides you through building a production-grade MLOps system for energy demand forecasting. Each module builds on the previous one, progressing from data engineering fundamentals to full CI/CD and orchestration.

**Estimated Total Duration:** 12-16 hours (self-paced)

---

## Module 1: Data Engineering

**Duration:** 1.5 - 2 hours

### Learning Objectives

- Understand the structure of time-series energy consumption data
- Generate realistic synthetic data with seasonal patterns and weather correlations
- Validate data quality using declarative expectation suites
- Build a simple feature store for reproducible feature engineering

### Exercises

1. **Explore the Synthetic Data Generator**
   - Read through `src/energy_forecast/data/synthetic.py`
   - Generate data for 10 buildings over 1 year
   - Visualize daily, weekly, and yearly seasonality patterns using the provided notebooks
   - Experiment with different `SeasonalWeights` and `WeatherParams` configurations

2. **Implement Data Validation**
   - Review the validation rules in `src/energy_forecast/data/validator.py`
   - Add a new validation rule: energy consumption should not exceed 3 standard deviations from the building mean
   - Introduce intentional data quality issues and verify the validator catches them

3. **Build the Feature Pipeline**
   - Run the data processor to clean and impute missing values
   - Generate temporal features (hour, day of week, month, cyclical encodings)
   - Generate lag features (1h, 24h, 168h lookback)
   - Generate rolling statistics (24h mean, 24h standard deviation)
   - Save the feature matrix to the feature store at `data/features/`

4. **Data Exploration Notebook**
   - Open `notebooks/` and explore the data distributions
   - Plot the correlation matrix between features and energy consumption
   - Identify the most predictive features

### Expected Outcomes

- A Parquet file at `data/raw/energy_data.parquet` with 50 buildings and 4 years of hourly data
- A validated, feature-engineered dataset saved in the feature store
- An understanding of how seasonal patterns, weather, and building profiles affect energy consumption

---

## Module 2: Experiment Tracking

**Duration:** 1.5 - 2 hours

### Learning Objectives

- Set up and navigate the MLflow tracking server
- Log parameters, metrics, and artifacts from training runs
- Compare experiments across different model configurations
- Understand the MLflow Model Registry lifecycle

### Exercises

1. **MLflow Setup and Tour**
   - Access MLflow UI at http://localhost:5000
   - Create a new experiment called `my-first-experiment`
   - Use the MLflow Python API to log a simple run with parameters and metrics

2. **Log a Training Run**
   - Train a Ridge regression model with different alpha values (0.01, 0.1, 1.0, 10.0)
   - Log each run to MLflow with: hyperparameters, train/val metrics, feature importance plot
   - Add tags to distinguish runs (e.g., `model_type`, `data_version`)

3. **Compare Experiments**
   - Use the MLflow comparison view to find the best alpha value
   - Sort by RMSE and MAE to understand the trade-off
   - Download artifacts from the best run

4. **Model Registry**
   - Register the best model in the MLflow Model Registry
   - Transition it through stages: None -> Staging -> Production
   - Understand what happens when you promote a new version to Production

5. **Artifact Storage**
   - Open the MinIO console at http://localhost:9001
   - Browse the `mlflow` bucket to see how artifacts are stored
   - Understand the relationship between MLflow run IDs and MinIO object paths

### Expected Outcomes

- Multiple experiment runs logged to MLflow with full metadata
- A registered model in the Model Registry with version history
- Understanding of how MLflow, PostgreSQL, and MinIO work together

---

## Module 3: Model Training

**Duration:** 2 - 3 hours

### Learning Objectives

- Implement and train multiple model architectures (linear, tree-based, deep learning)
- Use time-series cross-validation to avoid data leakage
- Perform hyperparameter optimization with Optuna
- Evaluate and compare model performance fairly

### Exercises

1. **Baseline Model**
   - Train a Ridge regression model using `src/energy_forecast/models/linear.py`
   - Evaluate on the validation set and record RMSE, MAE, MAPE, and R-squared
   - This establishes the performance floor that more complex models must beat

2. **XGBoost Model**
   - Train XGBoost using the default configuration in `configs/model_config.yaml`
   - Enable early stopping and monitor the validation loss curve
   - Examine feature importance: which features contribute most to predictions?
   - Compare against the baseline -- what is the RMSE improvement?

3. **LSTM Model** (optional, requires more compute)
   - Review the LSTM architecture in `src/energy_forecast/models/lstm_model.py`
   - Prepare sequence data: 168-hour (1 week) sliding windows
   - Train for 50 epochs with early stopping (patience=10)
   - Compare against XGBoost on the same validation set

4. **Hyperparameter Tuning**
   - Use the Optuna integration in `src/energy_forecast/training/hyperparameter.py`
   - Run 20 trials of XGBoost hyperparameter search
   - Examine the Optuna optimization history: which parameters matter most?
   - Train the final model with the best hyperparameters

5. **Time-Series Cross-Validation**
   - Run 5-fold expanding window cross-validation
   - Understand why random K-fold is inappropriate for time-series data
   - Observe how the 24-hour gap prevents lookahead bias
   - Report mean and standard deviation of metrics across folds

### Expected Outcomes

- Three trained models (Linear, XGBoost, LSTM) logged to MLflow
- A hyperparameter-tuned XGBoost model with documented best parameters
- Cross-validation results demonstrating model stability
- Clear understanding of which model performs best and why

---

## Module 4: Model Serving

**Duration:** 1.5 - 2 hours

### Learning Objectives

- Deploy a trained model as a REST API using FastAPI
- Understand request validation, feature engineering at inference time, and response formatting
- Implement batch prediction for high-throughput scenarios
- Test the API with realistic requests

### Exercises

1. **Explore the API Architecture**
   - Read through `src/energy_forecast/serving/app.py` (application factory)
   - Understand the model loading lifecycle (lifespan context manager)
   - Review request/response schemas in `src/energy_forecast/serving/schemas.py`
   - Examine the middleware stack (logging, CORS)

2. **Start the API and Test**
   - Start the API server: `make serve`
   - Open the Swagger docs at http://localhost:8000/docs
   - Make a single prediction using the interactive docs
   - Make the same prediction using curl

3. **Batch Predictions**
   - Send a batch request with 10 buildings for the same timestamp
   - Send a batch request with 24 timestamps for the same building (full day forecast)
   - Measure the processing time and compare with 24 individual requests

4. **Error Handling**
   - Send a request with missing required fields -- observe the 422 response
   - Send a request with humidity > 100 -- observe the validation error
   - Stop the model from being loaded and observe the 503 degraded health check

5. **Feature Vector Construction**
   - Read through the `_build_feature_vector` function in `routes.py`
   - Understand how the timestamp is encoded (sin/cos cyclical features)
   - Add a new optional feature to the request schema and trace it through to prediction

### Expected Outcomes

- A running prediction API serving the Production model from MLflow
- Experience with REST API design patterns for ML serving
- Understanding of how inference-time feature engineering differs from training

---

## Module 5: Monitoring and Observability

**Duration:** 2 - 2.5 hours

### Learning Objectives

- Detect data drift and concept drift in production predictions
- Set up Prometheus metrics collection for ML-specific signals
- Build Grafana dashboards for model performance monitoring
- Understand the feedback loop between monitoring and retraining

### Exercises

1. **Prometheus Metrics**
   - Review `src/energy_forecast/monitoring/prometheus_metrics.py`
   - Make 50+ predictions and then visit Prometheus at http://localhost:9090
   - Query `prediction_latency_seconds` to see latency distribution
   - Query `prediction_requests_total` to see request counts by status

2. **Grafana Dashboards**
   - Log in to Grafana at http://localhost:3000 (admin/admin)
   - Explore the pre-provisioned dashboards
   - Create a new panel: prediction latency 95th percentile over time
   - Create a new panel: prediction value distribution histogram
   - Set up an alert: notify if p99 latency exceeds 500ms

3. **Drift Detection**
   - Review `src/energy_forecast/monitoring/drift.py`
   - Generate a reference dataset (training data distribution)
   - Simulate drift: send predictions with temperatures shifted by 10 degrees
   - Run the drift detector and examine the Evidently report
   - Understand which statistical tests are used (KS test, PSI)

4. **Performance Monitoring**
   - Review `src/energy_forecast/monitoring/performance.py`
   - Track model accuracy over time using ground truth data
   - Simulate concept drift: what happens when the relationship between features and energy changes?
   - Set up an alert for accuracy degradation

5. **Design the Feedback Loop**
   - Sketch the complete monitoring-to-retraining loop
   - Define thresholds: at what drift score should retraining trigger?
   - Discuss trade-offs: retraining too often vs. too rarely

### Expected Outcomes

- A working Grafana dashboard monitoring API performance and model health
- Experience running drift detection and interpreting results
- A design document for automated retraining triggered by monitoring signals

---

## Module 6: CI/CD and Deployment

**Duration:** 1.5 - 2 hours

### Learning Objectives

- Set up continuous integration for an ML project
- Build and publish Docker images for the prediction service
- Deploy to Kubernetes with health checks and autoscaling
- Understand the ML-specific considerations in CI/CD pipelines

### Exercises

1. **CI Pipeline Review**
   - Read through `.github/workflows/ci.yml`
   - Understand the three stages: lint, type-check, test
   - Run the pipeline locally: `make lint && make type-check && make test`
   - Add a new test and verify it passes in CI

2. **Docker Build**
   - Examine the `Dockerfile` and understand the multi-stage build
   - Build the image locally: `docker build -t energy-forecast-api .`
   - Run the container and verify the API responds
   - Understand layer caching and how to optimize build times

3. **Kubernetes Deployment**
   - Review the manifests in `deployment/kubernetes/`:
     - `namespace.yaml` -- isolated namespace
     - `configmap.yaml` -- environment configuration
     - `deployment.yaml` -- pod spec with health checks
     - `service.yaml` -- ClusterIP service
     - `hpa.yaml` -- Horizontal Pod Autoscaler
   - Understand readiness vs. liveness probes
   - Understand HPA scaling based on CPU utilization

4. **Deployment Pipeline**
   - Review `.github/workflows/deploy.yml`
   - Understand the deployment stages: build, push, deploy
   - Discuss blue-green vs. rolling deployment strategies for ML services
   - What happens to in-flight requests during a model version update?

5. **ML-Specific CI Considerations**
   - How do you test model quality in CI (not just code quality)?
   - Discuss model validation gates: minimum accuracy, no regression
   - When should training run: in CI, scheduled, or triggered by drift?

### Expected Outcomes

- Understanding of the full CI/CD pipeline from commit to deployment
- Ability to read and modify GitHub Actions workflows
- Knowledge of Kubernetes deployment patterns for ML services

---

## Module 7: Orchestration

**Duration:** 1.5 - 2 hours

### Learning Objectives

- Build and manage Airflow DAGs for ML pipelines
- Schedule periodic retraining with proper dependency management
- Handle failures, retries, and alerting in production pipelines
- Understand the relationship between orchestration and the rest of the MLOps stack

### Exercises

1. **Airflow Setup and Tour**
   - Access Airflow at http://localhost:8080 (admin/admin)
   - Find the `energy_demand_training` DAG
   - Examine the DAG graph view to see task dependencies
   - Trigger a manual DAG run and monitor its progress

2. **Understand the Training DAG**
   - Read through `airflow/dags/training_dag.py`
   - Map each task to its corresponding Python function
   - Understand the task dependency chain:
     generate_data -> validate_data -> engineer_features -> train_model -> evaluate_model -> register_model

3. **DAG Configuration**
   - Modify the schedule to run daily instead of weekly
   - Add a new task: send a Slack/email notification after training completes
   - Add a branching task: only register the model if RMSE is below a threshold
   - Test with `airflow dags test energy_demand_training 2024-01-01`

4. **Failure Handling**
   - Introduce an intentional failure in the validation task
   - Observe the retry behavior (2 retries, 10-minute delay)
   - Check the Airflow logs for error details
   - Clear the failed task and re-run

5. **Build a New DAG**
   - Create a DAG for batch inference:
     - Load the latest data
     - Run batch predictions using the Production model
     - Save results to `data/processed/predictions.parquet`
     - Generate a drift report
   - Schedule it to run hourly

### Expected Outcomes

- A working understanding of Airflow DAGs, operators, and scheduling
- Experience debugging failed tasks and managing retries
- A new batch inference DAG running alongside the training DAG

---

## Progression Map

```
Module 1          Module 2          Module 3
Data Engineering  Experiment        Model Training
     |            Tracking               |
     |                |                  |
     v                v                  v
Module 4 -------> Module 5 -------> Module 6
Model Serving     Monitoring        CI/CD &
                  & Observability   Deployment
                       |
                       v
                  Module 7
                  Orchestration
```

Modules 1-3 can be completed somewhat independently, but each builds knowledge for Modules 4-7. Module 5 (Monitoring) is the linchpin that connects serving to retraining.
