"""Tests for config loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from energy_forecast.utils.config import Settings, load_config

pytestmark = pytest.mark.unit


class TestConfig:
    """Tests for config loading functions."""

    def test_load_config_returns_dict(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_file = config_dir / "test_config.yaml"
        config_file.write_text(yaml.dump({"key": "value", "nested": {"a": 1}}))

        monkeypatch.setenv("ENERGY_FORECAST_CONFIGS_DIR", str(config_dir))
        result = load_config("test_config")
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_load_config_missing_file_raises(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "empty_configs"
        config_dir.mkdir()
        monkeypatch.setenv("ENERGY_FORECAST_CONFIGS_DIR", str(config_dir))
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config")

    def test_settings_defaults(self):
        settings = Settings()
        assert settings.mlflow_tracking_uri == "http://localhost:5000"
        assert settings.api_port == 8000
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.postgres_port == 5432
