"""Unit tests for prediction logger."""
from unittest.mock import MagicMock

from src.ml.serving.prediction_logger import PredictionLogger


def test_buffer_flush_on_size():
    """Buffer should flush when it reaches flush_size."""
    mock_s3 = MagicMock()
    pl = PredictionLogger(s3_client=mock_s3, flush_size=2, flush_interval_s=9999)

    # First record: no flush yet
    pl.log(
        {"timestamp_brussels": "2024-01-01T12:00:00", "temperature_2m": 10.0},
        {"predicted_load_mw": 9000.0, "prediction_id": "abc", "model_version": "1", "predicted_at": "2024-01-01"},
    )
    assert mock_s3.put_object.call_count == 0

    # Second record: triggers flush
    pl.log(
        {"timestamp_brussels": "2024-01-01T13:00:00", "temperature_2m": 11.0},
        {"predicted_load_mw": 9100.0, "prediction_id": "def", "model_version": "1", "predicted_at": "2024-01-01"},
    )
    assert mock_s3.put_object.call_count == 1

    pl.close()


def test_close_flushes_remaining():
    """Closing the logger should flush any remaining records."""
    mock_s3 = MagicMock()
    pl = PredictionLogger(s3_client=mock_s3, flush_size=100, flush_interval_s=9999)

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
    pl = PredictionLogger(s3_client=mock_s3, flush_size=1, flush_interval_s=9999)

    pl.log(
        {"timestamp_brussels": "2024-01-01T12:00:00"},
        {"predicted_load_mw": 9000.0, "prediction_id": "abc", "model_version": "1", "predicted_at": "2024-01-01"},
    )

    # Buffer should still have the record after failed flush
    assert len(pl._buffer) == 1
    pl._timer_running = False
