# API Reference

The Energy Demand Forecasting API is a RESTful service built with FastAPI. It provides real-time and batch energy demand predictions for buildings.

---

## Base URL

```
http://localhost:8000
```

When running in Docker, the API binds to port 8000 by default. Change this via the `API_PORT` environment variable.

## Interactive Documentation

FastAPI auto-generates interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Authentication

This system does not require authentication. It is designed as a workshop/learning environment. In a production deployment, you would add API key or OAuth2 authentication via FastAPI's dependency injection.

---

## Endpoints

### POST /predict

Make a single energy demand prediction for a building at a specific timestamp.

**Request Body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `timestamp` | string (ISO-8601) | Yes | Target forecast timestamp |
| `building_id` | string | Yes | Unique building identifier |
| `temperature` | float | Yes | Outdoor temperature in degrees Celsius |
| `humidity` | float (0-100) | Yes | Relative humidity as a percentage |
| `features` | object | No | Additional key-value features |

**Example Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T14:00:00",
    "building_id": "BLDG_001",
    "temperature": 22.5,
    "humidity": 45.0,
    "features": {
      "occupancy_rate": 0.85,
      "is_holiday": 0
    }
  }'
```

**Example Response (200 OK):**

```json
{
  "building_id": "BLDG_001",
  "timestamp": "2024-01-15T14:00:00",
  "predicted_kwh": 187.4523,
  "confidence_lower": 168.707,
  "confidence_upper": 206.1976,
  "model_version": "3"
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `building_id` | string | The requested building ID |
| `timestamp` | string | The forecast timestamp |
| `predicted_kwh` | float | Predicted energy demand in kilowatt-hours |
| `confidence_lower` | float | Lower bound of confidence interval |
| `confidence_upper` | float | Upper bound of confidence interval |
| `model_version` | string | Version of the model that produced the prediction |

---

### POST /predict/batch

Make predictions for multiple buildings or timestamps in a single request.

**Request Body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `predictions` | array | Yes | List of prediction request objects (minimum 1) |

Each element in the `predictions` array follows the same schema as the `/predict` request body.

**Example Request:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "timestamp": "2024-01-15T14:00:00",
        "building_id": "BLDG_001",
        "temperature": 22.5,
        "humidity": 45.0
      },
      {
        "timestamp": "2024-01-15T14:00:00",
        "building_id": "BLDG_002",
        "temperature": 22.5,
        "humidity": 45.0
      },
      {
        "timestamp": "2024-01-15T15:00:00",
        "building_id": "BLDG_001",
        "temperature": 23.1,
        "humidity": 43.0
      }
    ]
  }'
```

**Example Response (200 OK):**

```json
{
  "predictions": [
    {
      "building_id": "BLDG_001",
      "timestamp": "2024-01-15T14:00:00",
      "predicted_kwh": 187.4523,
      "confidence_lower": 168.707,
      "confidence_upper": 206.1976,
      "model_version": "3"
    },
    {
      "building_id": "BLDG_002",
      "timestamp": "2024-01-15T14:00:00",
      "predicted_kwh": 142.8901,
      "confidence_lower": 128.6011,
      "confidence_upper": 157.1791,
      "model_version": "3"
    },
    {
      "building_id": "BLDG_001",
      "timestamp": "2024-01-15T15:00:00",
      "predicted_kwh": 191.2037,
      "confidence_lower": 172.0833,
      "confidence_upper": 210.3241,
      "model_version": "3"
    }
  ],
  "processing_time_ms": 12.45
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `predictions` | array | List of individual prediction responses |
| `processing_time_ms` | float | Total batch processing time in milliseconds |

---

### GET /health

Check the health status of the API service and model readiness.

**Example Request:**

```bash
curl http://localhost:8000/health
```

**Example Response (200 OK):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3",
  "uptime_seconds": 3621.45
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"healthy"` if model is loaded, `"degraded"` otherwise |
| `model_loaded` | boolean | Whether a model is currently available for inference |
| `model_version` | string or null | Version of the loaded model, null if no model loaded |
| `uptime_seconds` | float | Seconds since the API process started |

---

### GET /model/info

Retrieve metadata about the currently loaded model.

**Example Request:**

```bash
curl http://localhost:8000/model/info
```

**Example Response (200 OK):**

```json
{
  "model_name": "energy-demand-forecaster",
  "model_version": "3",
  "model_stage": "Production",
  "metrics": {
    "rmse": 12.45,
    "mae": 8.73,
    "mape": 4.21,
    "r2": 0.94
  },
  "features_used": [
    "temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "month",
    "is_weekend",
    "energy_lag_24h",
    "energy_rolling_mean_24h"
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `model_name` | string | Registered model name in MLflow |
| `model_version` | string | Model version number |
| `model_stage` | string | MLflow stage (Production, Staging, Archived) |
| `metrics` | object | Training/validation metrics for this model version |
| `features_used` | array | List of feature names the model expects |

---

## Error Responses

All errors return a consistent JSON envelope:

```json
{
  "error": "Error category",
  "detail": "Human-readable description of what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning | When It Occurs |
|---|---|---|
| `200` | OK | Successful request |
| `422` | Unprocessable Entity | Request body fails Pydantic validation (missing fields, wrong types, humidity out of range) |
| `500` | Internal Server Error | Unexpected error during prediction |
| `503` | Service Unavailable | No model is loaded (startup in progress or model load failed) |

### Validation Error Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"building_id": "BLDG_001"}'
```

Response (422):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "timestamp"],
      "msg": "Field required",
      "input": {"building_id": "BLDG_001"}
    },
    {
      "type": "missing",
      "loc": ["body", "temperature"],
      "msg": "Field required",
      "input": {"building_id": "BLDG_001"}
    },
    {
      "type": "missing",
      "loc": ["body", "humidity"],
      "msg": "Field required",
      "input": {"building_id": "BLDG_001"}
    }
  ]
}
```

### Service Unavailable Example

If the model has not been trained and registered yet:

```json
{
  "error": "Service Unavailable",
  "detail": "No model is currently loaded."
}
```

---

## Rate Limiting

There is no rate limiting configured by default in this workshop environment. For production deployments, you would add rate limiting via:

- **Reverse proxy** (nginx, Traefik) -- recommended for most use cases
- **FastAPI middleware** (e.g., `slowapi`) -- application-level limiting
- **API gateway** (Kong, AWS API Gateway) -- for cloud deployments

A reasonable starting point for production rate limits:

| Endpoint | Suggested Limit |
|---|---|
| `POST /predict` | 100 requests/second per client |
| `POST /predict/batch` | 10 requests/second per client |
| `GET /health` | 60 requests/second per client |
| `GET /model/info` | 30 requests/second per client |

---

## CORS

Cross-Origin Resource Sharing is enabled for the following origins by default:

- `http://localhost:3000` (Grafana)
- `http://localhost:8080` (Airflow)

Additional origins can be configured in `configs/serving_config.yaml`.

---

## Prometheus Metrics

The API exposes Prometheus-compatible metrics that are scraped automatically:

| Metric | Type | Description |
|---|---|---|
| `prediction_latency_seconds` | Histogram | Time to compute a single prediction |
| `prediction_value_kwh` | Histogram | Distribution of predicted values |
| `prediction_requests_total` | Counter | Total predictions by status and model version |

These metrics power the Grafana dashboards for API performance monitoring.
