#!/bin/bash
set -euo pipefail
echo "=== Energy Demand Forecasting MLOps - Infrastructure Setup ==="

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# Create required directories
mkdir -p data/raw data/processed data/features logs

# Build and start Docker services
echo "Starting Docker services..."
docker compose up -d --build

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "=== Services Started ==="
echo "MLflow UI:        http://localhost:5000"
echo "FastAPI Docs:     http://localhost:8000/docs"
echo "Airflow UI:       http://localhost:8080"
echo "Prometheus:       http://localhost:9090"
echo "Grafana:          http://localhost:3000"
echo "MinIO Console:    http://localhost:9001"
echo ""
echo "Default credentials:"
echo "  Airflow: admin / admin"
echo "  Grafana: admin / admin"
echo "  MinIO:   minioadmin / minioadmin"
