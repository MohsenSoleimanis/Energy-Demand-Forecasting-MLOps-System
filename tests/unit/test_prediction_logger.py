"""Unit tests for prediction logger."""
from unittest.mock import MagicMock

import pytest

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow required for parquet tests")

from src.ml.serving.prediction_logger import PredictionLogger

BUCKET = "test-bucket"
PREFIX = "test-predictions"


def _make_logger(mock_s3: MagicMock, **kwargs) -> PredictionLogger:
    """Create a PredictionLogger with test defaults."""
    defaults = {
        "s3_client": mock_s3,
        "bucket": BUCKET,
        "prefix": PREFIX,
        "flush_interval_s": 9999,
    }
    defaults.update(kwargs)
    return PredictionLogger(**defaults)


def test_buffer_flush_on_size():
    """Buffer should flush when it reaches flush_size."""
    mock_s3 = MagicMock()
    pl = _make_logger(mock_s3, flush_size=2)

    pl.log(
        {"timestamp_brussels": "2024-01-01T12:00:00", "temperature_2m": 10.0},
        {"predicted_load_mw": 9000.0, "prediction_id": "abc", "model_version": "1", "predicted_at": "2024-01-01"},
    )
    assert mock_s3.put_object.call_count == 0

    pl.log(
        {"timestamp_brussels": "2024-01-01T13:00:00", "temperature_2m": 11.0},
        {"predicted_load_mw": 9100.0, "prediction_id": "def", "model_version": "1", "predicted_at": "2024-01-01"},
    )
    assert mock_s3.put_object.call_count == 1

    pl.close()


def test_close_flushes_remaining():
    """Closing the logger should flush any remaining records."""
    mock_s3 = MagicMock()
    pl = _make_logger(mock_s3, flush_size=100)

    pl.log(
        {"timestamp_brussels": "2024-01-01T12:00:00"},
        {"predicted_load_mw": 9000.0, "prediction_id": "abc", "model_version": "1", "predicted_at": "2024-01-01"},
    )
    assert mock_s3.put_object.call_count == 0

    pl.close()
    assert mock_s3.put_object.call_count == 1


def test_flush_failure_preserves_records():
    """If S3 write fails, records should be preserved in buffer."""
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = Exception("S3 error")
    pl = _make_logger(mock_s3, flush_size=1)

    pl.log(
        {"timestamp_brussels": "2024-01-01T12:00:00"},
        {"predicted_load_mw": 9000.0, "prediction_id": "abc", "model_version": "1", "predicted_at": "2024-01-01"},
    )

    assert len(pl._buffer) == 1
    pl._timer_running = False
