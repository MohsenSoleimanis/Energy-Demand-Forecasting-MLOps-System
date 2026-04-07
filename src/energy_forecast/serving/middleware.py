"""Request logging and Prometheus metrics middleware."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from energy_forecast.monitoring.prometheus_metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
)

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every HTTP request and updates Prometheus counters.

    For each request the middleware:

    * Generates a unique ``X-Request-ID`` header.
    * Records the wall-clock latency.
    * Emits a structured log line.
    * Increments ``http_requests_total`` and observes
      ``http_request_duration_seconds``.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attach request ID so downstream handlers can access it.
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            _record(request, 500, latency_ms, request_id)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        _record(request, response.status_code, latency_ms, request_id)

        response.headers["X-Request-ID"] = request_id
        return response


def _record(
    request: Request,
    status_code: int,
    latency_ms: float,
    request_id: str,
) -> None:
    """Log the request and push Prometheus observations."""
    method = request.method
    path = request.url.path

    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )

    REQUEST_COUNT.labels(method=method, path=path, status_code=str(status_code)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(latency_ms / 1000)
