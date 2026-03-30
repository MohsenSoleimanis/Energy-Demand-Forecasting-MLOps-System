"""Register the latest trained model as production in MLflow."""
import os

import mlflow

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.MlflowClient()

# Find latest run
runs = client.search_runs("1", order_by=["start_time DESC"], max_results=1)
if not runs:
    print("No runs found. Train a model first: python run.py train")
    exit(1)

run = runs[0]
print(f"Latest run: {run.info.run_id}")
print(f"  MAPE: {run.data.metrics.get('test_mape', 'N/A')}")
print(f"  MAE:  {run.data.metrics.get('test_mae', 'N/A')}")
print(f"  R2:   {run.data.metrics.get('test_r2', 'N/A')}")

# Register model
mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "energy-demand-forecast")
print(f"\nRegistered model version {mv.version}")

# Set as production
client.set_registered_model_alias("energy-demand-forecast", "production", mv.version)
print(f"Set version {mv.version} as 'production'")
print("\nDone. The API should now return 'healthy' at http://localhost:8000/health")
