"""Unit tests for alerting module (MON-004)."""

import logging
import pytest

from src.ml.monitoring.alerting import AlertLevel, classify_alert, Alert, send_alert


class TestClassifyAlert:
    def test_critical_on_pipeline_failure(self):
        level = classify_alert(pipeline_failure=True)
        assert level == AlertLevel.CRITICAL

    def test_critical_on_high_mape(self):
        level = classify_alert(mape=0.12)
        assert level == AlertLevel.CRITICAL

    def test_warning_on_moderate_mape(self):
        level = classify_alert(mape=0.08)
        assert level == AlertLevel.WARNING

    def test_warning_on_high_psi(self):
        level = classify_alert(max_psi=0.25)
        assert level == AlertLevel.WARNING

    def test_info_on_low_metrics(self):
        level = classify_alert(mape=0.03, max_psi=0.05)
        assert level == AlertLevel.INFO

    def test_critical_overrides_warning(self):
        level = classify_alert(mape=0.15, max_psi=0.25)
        assert level == AlertLevel.CRITICAL

    def test_none_values(self):
        level = classify_alert()
        assert level == AlertLevel.INFO


class TestSendAlert:
    def test_send_alert_logs(self, caplog):
        alert = Alert(
            level=AlertLevel.WARNING,
            source="test",
            message="Test warning",
            details={"mape": 0.08},
        )
        with caplog.at_level(logging.WARNING):
            send_alert(alert)
        assert "Test warning" in caplog.text
