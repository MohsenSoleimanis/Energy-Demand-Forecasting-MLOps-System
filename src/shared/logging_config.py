"""Structured logging configuration for the Belgian Energy Demand Forecasting system.

Provides a JSON formatter for production environments and a human-readable
formatter for local development.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Suitable for structured-log aggregation systems (ELK, CloudWatch, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON-encoded string for *record*.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string.
        """
        entry: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure the root logger.

    Args:
        level: Log level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        json_format: If ``True`` use :class:`JSONFormatter` (production).
            If ``False`` use a plain-text format (development).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        )

    root.handlers.clear()
    root.addHandler(handler)
