"""Configuration loading with YAML files and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

# Resolve the project root: walk up from this file until we find pyproject.toml
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR
for _ in range(10):
    if (_PROJECT_ROOT / "pyproject.toml").exists():
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent
else:
    # Fallback: assume three levels up from utils/config.py -> src/energy_forecast/utils
    _PROJECT_ROOT = _THIS_DIR.parent.parent.parent

CONFIGS_DIR = _PROJECT_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_name: str, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load a YAML config file from the ``configs/`` directory.

    Parameters
    ----------
    config_name:
        File name (with or without ``.yaml`` extension) inside ``configs/``.
    overrides:
        Optional dictionary of values to merge on top of the loaded config.

    Returns
    -------
    dict
        The parsed configuration dictionary with any overrides applied.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    if not config_name.endswith((".yaml", ".yml")):
        config_name = f"{config_name}.yaml"

    config_path = CONFIGS_DIR / config_name

    # Allow an env var to redirect the configs directory
    env_configs_dir = os.getenv("ENERGY_FORECAST_CONFIGS_DIR")
    if env_configs_dir:
        config_path = Path(env_configs_dir) / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    if overrides:
        config = _deep_merge(config, overrides)

    return config


class Settings(BaseSettings):
    """Application-wide settings populated from environment variables.

    Environment variables take precedence; sensible defaults are provided so the
    application can start without any env vars set during local development.
    """

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI.",
    )
    mlflow_experiment_name: str = Field(
        default="energy-demand-forecasting",
        description="Default MLflow experiment name.",
    )

    # --- PostgreSQL ---
    postgres_host: str = Field(default="localhost", description="PostgreSQL host.")
    postgres_port: int = Field(default=5432, description="PostgreSQL port.")
    postgres_user: str = Field(default="energy_forecast", description="PostgreSQL user.")
    postgres_password: str = Field(default="energy_forecast", description="PostgreSQL password.")
    postgres_db: str = Field(default="energy_forecast", description="PostgreSQL database name.")

    # --- API / Serving ---
    api_host: str = Field(default="0.0.0.0", description="API server bind host.")
    api_port: int = Field(default=8000, description="API server bind port.")
    api_workers: int = Field(default=4, description="Number of API worker processes.")

    # --- General ---
    log_level: str = Field(default="INFO", description="Root log level.")
    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production).",
    )
    data_dir: str = Field(default="data", description="Root data directory.")

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- Derived helpers ---

    @property
    def postgres_dsn(self) -> str:
        """Return a PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
