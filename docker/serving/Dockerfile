# INFRA-005: Serving Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    mlflow \
    lightgbm \
    pandas \
    numpy \
    prometheus-client \
    boto3 \
    hydra-core

COPY src/ml/__init__.py src/ml/__init__.py
COPY src/ml/serving/ src/ml/serving/
COPY src/ml/features/ src/ml/features/
COPY configs/serving/ configs/serving/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.ml.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
