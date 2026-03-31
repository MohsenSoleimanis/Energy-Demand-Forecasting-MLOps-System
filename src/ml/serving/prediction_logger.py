"""Buffered prediction logger that writes to MinIO/S3 as Parquet.

All configuration (bucket, prefix, flush parameters, retry limit)
is injected via the constructor -- no hardcoded values.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Thread-safe buffered logger that writes predictions to S3 as Parquet.

    Predictions are accumulated in memory and flushed to S3 either when
    the buffer reaches *flush_size* records or when *flush_interval_s*
    seconds have elapsed since the last flush.

    Parquet files are partitioned by date:
    ``s3://<bucket>/<prefix>/date=YYYY-MM-DD/batch_HHMMSS_ffffff.parquet``

    Args:
        s3_client: A configured boto3 S3 client.
        bucket: Target S3 bucket name.
        prefix: Object key prefix for prediction logs.
        flush_size: Number of buffered records that triggers a flush.
        flush_interval_s: Seconds between time-based flushes.
        max_retry_count: Maximum consecutive flush failures before
            records are dropped to prevent unbounded memory growth.
    """

    def __init__(
        self,
        s3_client: S3Client,
        bucket: str,
        prefix: str,
        flush_size: int = 50,
        flush_interval_s: float = 60.0,
        max_retry_count: int = 3,
    ) -> None:
        self._s3: S3Client = s3_client
        self._bucket: str = bucket
        self._prefix: str = prefix
        self._flush_size: int = flush_size
        self._flush_interval_s: float = flush_interval_s
        self._max_retry_count: int = max_retry_count

        self._buffer: list[dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()
        self._last_flush: float = time.monotonic()
        self._consecutive_failures: int = 0

        # Background flush timer
        self._timer_running: bool = True
        self._timer: threading.Thread = threading.Thread(
            target=self._flush_timer, daemon=True
        )
        self._timer.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, request_data: dict[str, Any], response_data: dict[str, Any]) -> None:
        """Add a prediction record to the buffer.  Thread-safe.

        Args:
            request_data: Serialized ``PredictionRequest``.
            response_data: Serialized ``PredictionResponse`` (may include
                shadow prediction metadata).
        """
        record: dict[str, Any] = {
            **{
                f"input_{k}": v
                for k, v in request_data.items()
                if k != "timestamp_brussels"
            },
            "timestamp_brussels": request_data.get("timestamp_brussels"),
            "predicted_load_mw": response_data.get("predicted_load_mw"),
            "prediction_id": response_data.get("prediction_id"),
            "model_version": response_data.get("model_version"),
            "predicted_at": response_data.get("predicted_at"),
        }

        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._flush_size:
                self._flush()

    def close(self) -> None:
        """Flush remaining records and stop the background timer."""
        self._timer_running = False
        with self._lock:
            self._flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_timer(self) -> None:
        """Background thread that triggers time-based flushes."""
        while self._timer_running:
            time.sleep(5)  # Check every 5 seconds
            with self._lock:
                elapsed: float = time.monotonic() - self._last_flush
                if self._buffer and elapsed >= self._flush_interval_s:
                    self._flush()

    def _flush(self) -> None:
        """Write buffered records to S3 as Parquet.  Must hold *self._lock*."""
        if not self._buffer:
            return

        records: list[dict[str, Any]] = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.monotonic()

        try:
            df: pd.DataFrame = pd.DataFrame(records)

            # Convert timestamp columns
            for col in ("timestamp_brussels", "predicted_at"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            # Partition by date of prediction
            now: datetime = datetime.now(UTC)
            date_str: str = now.strftime("%Y-%m-%d")
            batch_id: str = now.strftime("%H%M%S_%f")

            key: str = f"{self._prefix}/date={date_str}/batch_{batch_id}.parquet"

            buffer: io.BytesIO = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)

            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=buffer.getvalue(),
            )
            logger.info(
                "Flushed %d predictions to s3://%s/%s",
                len(records),
                self._bucket,
                key,
            )
            self._consecutive_failures = 0

        except Exception:
            self._consecutive_failures += 1
            logger.exception(
                "Failed to flush predictions to S3 (attempt %d/%d)",
                self._consecutive_failures,
                self._max_retry_count,
            )
            if self._consecutive_failures < self._max_retry_count:
                # Re-add records so they are retried on the next flush
                self._buffer = records + self._buffer
            else:
                logger.error(
                    "Dropping %d prediction records after %d consecutive "
                    "flush failures to prevent unbounded memory growth.",
                    len(records),
                    self._max_retry_count,
                )
                self._consecutive_failures = 0
