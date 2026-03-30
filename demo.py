"""
Quick demo script to run the system with synthetic data.
No API keys or external services needed - just Python + Docker.

Usage:
    python demo.py setup     # Generate synthetic data + train model
    python demo.py serve     # Start the API server
    python demo.py predict   # Send a test prediction
    python demo.py all       # Run everything
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def generate_synthetic_data():
    """Generate realistic synthetic Belgian energy data for demo purposes."""
    print("Generating synthetic training data...")

    n_hours = 365 * 24  # 1 year of hourly data
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_hours)]
    hours = np.array([t.hour for t in timestamps])
    months = np.array([t.month for t in timestamps])
    weekdays = np.array([t.weekday() for t in timestamps])

    np.random.seed(42)

    # Realistic Belgian load pattern: base + daily cycle + seasonal + noise
    daily_cycle = 2000 * np.sin(2 * np.pi * (hours - 6) / 24)
    seasonal_cycle = 1500 * np.cos(2 * np.pi * (months - 1) / 12)  # Higher in winter
    weekend_effect = -800 * (weekdays >= 5).astype(float)
    noise = np.random.normal(0, 300, n_hours)
    load = 9000 + daily_cycle + seasonal_cycle + weekend_effect + noise
    load = np.clip(load, 4500, 15000)

    # Temperature: seasonal + daily + noise
    temp_seasonal = 10 * np.cos(2 * np.pi * (months - 7) / 12)  # Warm in summer
    temp_daily = 5 * np.sin(2 * np.pi * (hours - 14) / 24)
    temperature = 12 + temp_seasonal + temp_daily + np.random.normal(0, 2, n_hours)

    df = pd.DataFrame({
        "timestamp_brussels": timestamps,
        "load_mw": load,
        "price_eur_mwh": 50 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 15, n_hours),
        "temperature_2m": temperature,
        "feels_like_temp": temperature - 2 + np.random.normal(0, 1, n_hours),
        "wind_speed_10m": np.abs(8 + np.random.normal(0, 5, n_hours)),
        "wind_direction_10m": np.random.uniform(0, 360, n_hours),
        "shortwave_radiation": np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 24)
                                          * (0.5 + 0.5 * np.sin(2 * np.pi * (months - 6) / 12))),
        "precipitation": np.random.exponential(0.5, n_hours),
        "cloud_cover": np.random.uniform(0, 100, n_hours),
        "pressure_msl": 1013 + np.random.normal(0, 8, n_hours),
        "renewable_share_pct": 20 + 15 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_hours),
        "nuclear_mw": 4000 + np.random.normal(0, 200, n_hours),
        "gas_mw": 2000 + 1000 * np.sin(2 * np.pi * (hours - 8) / 24) + np.random.normal(0, 200, n_hours),
        "is_belgian_holiday": [False] * n_hours,
        "is_weekend": [t.weekday() >= 5 for t in timestamps],
        "is_school_vacation": [t.month in (7, 8) for t in timestamps],
        "day_of_week": [t.weekday() for t in timestamps],
        "month": [t.month for t in timestamps],
        "hour_of_day": [t.hour for t in timestamps],
        "load_is_valid": [True] * n_hours,
        "weather_is_anomalous": [False] * n_hours,
    })

    # Add features
    from src.ml.features.feature_engineering import prepare_features
    df = prepare_features(df, mode="training")

    # Add target
    df["target_load_24h"] = df["load_mw"].shift(-24)

    # Filter: need 168h history and valid target
    df = df.iloc[168:-24].reset_index(drop=True)
    df = df.dropna(subset=["target_load_24h"])

    # Save
    output_dir = PROJECT_ROOT / "data" / "gold"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_set.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df)} rows to {output_path}")
    return df


def train_model(df=None):
    """Train a LightGBM model on synthetic data (no MLflow needed)."""
    print("Training LightGBM model...")

    if df is None:
        data_path = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
        df = pd.read_parquet(data_path)

    from src.ml.features.feature_engineering import get_feature_columns

    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    target_col = "target_load_24h"

    # Temporal split: 80% train, 10% val, 10% test
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = df.iloc[:train_end][feature_cols]
    y_train = df.iloc[:train_end][target_col]
    X_val = df.iloc[train_end:val_end][feature_cols]
    y_val = df.iloc[train_end:val_end][target_col]
    X_test = df.iloc[val_end:][feature_cols]
    y_test = df.iloc[val_end:][target_col]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Evaluate
    from sklearn.metrics import mean_absolute_error, r2_score

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test.values - y_pred) / y_test.values)))
    r2 = r2_score(y_test, y_pred)

    print(f"  Test MAE:  {mae:.1f} MW")
    print(f"  Test MAPE: {mape:.4f} ({mape*100:.2f}%)")
    print(f"  Test R2:   {r2:.4f}")

    # Save model locally
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    import pickle
    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols}, f)
    print(f"  Model saved to {model_path}")

    # Save metrics
    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics = {"test_mae": mae, "test_mape": mape, "test_r2": r2}
    with open(metrics_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, feature_cols


def start_demo_server():
    """Start a simple FastAPI server using the locally saved model."""
    import pickle

    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from src.ml.features.feature_engineering import prepare_features
    from src.ml.serving.schemas import (
        HealthResponse,
        PredictionRequest,
        PredictionResponse,
    )

    model_path = PROJECT_ROOT / "models" / "model.pkl"
    if not model_path.exists():
        print("No model found. Run 'python demo.py setup' first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    feature_cols = data["feature_columns"]

    app = FastAPI(title="Belgian Energy Demand Forecast - Demo")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="healthy", model_version="demo-1.0", model_alias="local")

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictionRequest):
        df = pd.DataFrame([req.model_dump()])
        df = prepare_features(df, mode="serving")
        # Use only columns the model expects, fill missing with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        prediction = model.predict(df[feature_cols])
        return PredictionResponse(
            timestamp_brussels=req.timestamp_brussels,
            predicted_load_mw=float(prediction[0]),
            model_version="demo-1.0",
        )

    print("\n  Starting demo API server...")
    print("  Health check: http://localhost:8000/health")
    print("  API docs:     http://localhost:8000/docs")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def send_test_prediction():
    """Send a test prediction to the running server."""
    import requests

    payload = {
        "timestamp_brussels": "2024-07-15T14:00:00",
        "temperature_2m": 25.0,
        "relative_humidity_2m": 55.0,
        "wind_speed_10m": 10.0,
        "wind_direction_10m": 200.0,
        "shortwave_radiation": 600.0,
        "precipitation": 0.0,
        "cloud_cover": 20.0,
        "pressure_msl": 1015.0,
    }

    try:
        resp = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        print("\nPrediction result:")
        print(f"  Timestamp:      {result['timestamp_brussels']}")
        print(f"  Predicted Load: {result['predicted_load_mw']:.1f} MW")
        print(f"  Model Version:  {result['model_version']}")
        print(f"  Prediction ID:  {result['prediction_id']}")
    except requests.ConnectionError:
        print("Error: Cannot connect to server. Run 'python demo.py serve' first.")
    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "setup":
        df = generate_synthetic_data()
        train_model(df)
        print("\nSetup complete! Now run: python demo.py serve")

    elif cmd == "serve":
        start_demo_server()

    elif cmd == "predict":
        send_test_prediction()

    elif cmd == "all":
        df = generate_synthetic_data()
        train_model(df)
        print("\nStarting server...")
        start_demo_server()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
