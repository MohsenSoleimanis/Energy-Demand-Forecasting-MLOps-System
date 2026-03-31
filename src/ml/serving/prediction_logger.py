"""Buffered prediction logger that writes to MinIO/S3 as Parquet."""
import io
import logging
import threading
import time
from datetime import UTC, datetime

import pandas as pd

logger = logging.getLogger(__name__)

BUCKET = "lakehouse"
PREFIX = "predictions"
DEFAULT_FLUSH_SIZE = 50
DEFAULT_FLUSH_INTERVAL_S = 60


class PredictionLogger:
    """Thread-safe buffered logger that writes predictions to S3 as Parquet.

    Predictions are accumulated in memory and flushed to S3 either when
    the buffer reaches `flush_size` records or `flush_interval_s` seconds
    have passed since the last flush.

    Parquet files are partitioned by date: s3://lakehouse/predictions/date=YYYY-MM-DD/
    """

    def __init__(
        self,
        s3_client,
        bucket: str = BUCKET,
        prefix: str = PREFIX,
        flush_size: int = DEFAULT_FLUSH_SIZE,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    ):
        self._s3 = s3_client
        self._bucket = bucket
        self._prefix = prefix
        self._flush_size = flush_size
        self._flush_interval_s = flush_interval_s
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

        # Start background flush timer
        self._timer_running = True
        self._timer = threading.Thread(target=self._flush_timer, daemon=True)
        self._timer.start()

    def log(self, request_data: dict, response_data: dict) -> None:
        """Add a prediction record to the buffer. Thread-safe."""
        record = {
            **{f"input_{k}": v for k, v in request_data.items()
               if k != "timestamp_brussels"},
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

    def _flush_timer(self):
        """Background thread that flushes on interval."""
        while self._timer_running:
            time.sleep(5)  # Check every 5 seconds
            with self._lock:
                elapsed = time.monotonic() - self._last_flush
                if self._buffer and elapsed >= self._flush_interval_s:
                    self._flush()

    def _flush(self):
        """Write buffered records to S3 as Parquet. Must hold self._lock."""
        if not self._buffer:
            return

        records = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.monotonic()

        try:
            df = pd.DataFrame(records)

            # Convert timestamp columns
            for col in ["timestamp_brussels", "predicted_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            # Partition by date of prediction
            now = datetime.now(UTC)
            date_str = now.strftime("%Y-%m-%d")
            batch_id = now.strftime("%H%M%S_%f")

            key = f"{self._prefix}/date={date_str}/batch_{batch_id}.parquet"

            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)

            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=buffer.getvalue(),
            )
            logger.info("Flushed %d predictions to s3://%s/%s", len(records), self._bucket, key)

        except Exception:
            logger.exception("Failed to flush predictions to S3")
            # Re-add records to buffer so they're not lost
            with self._lock:
                self._buffer = records + self._buffer

    def close(self):
        """Flush remaining records and stop the timer."""
        self._timer_running = False
        with self._lock:
            self._flush()
