"""Load test for the prediction API.

Run with::

    locust -f tests/load/locustfile.py --host http://localhost:8000

Simulates concurrent users sending prediction requests, health checks,
and metrics scrapes to validate API throughput and latency under load.
"""

from __future__ import annotations

from locust import HttpUser, between, task

API_KEY = "test-load-key"

SAMPLE_REQUEST = {
    "timestamp_brussels": "2026-03-31T14:00:00",
    "temperature_2m": 12.5,
    "relative_humidity_2m": 65.0,
    "wind_speed_10m": 15.0,
    "wind_direction_10m": 220.0,
    "shortwave_radiation": 350.0,
    "precipitation": 0.0,
    "cloud_cover": 40.0,
    "pressure_msl": 1018.0,
    "load_lag_24h": 9200.0,
    "load_lag_168h": 9500.0,
}


class ForecastUser(HttpUser):
    """Simulated user exercising the forecast API."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self) -> None:
        """Send a prediction request (highest frequency task)."""
        self.client.post(
            "/predict",
            json=SAMPLE_REQUEST,
            headers={"X-API-Key": API_KEY},
        )

    @task(1)
    def health(self) -> None:
        """Hit the health-check endpoint."""
        self.client.get("/health")

    @task(1)
    def metrics(self) -> None:
        """Scrape the Prometheus metrics endpoint."""
        self.client.get("/metrics")
