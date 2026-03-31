"""Centralized configuration loader.

All runtime values come from YAML config files or environment variables.
No hardcoded credentials, paths, or magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.shared.exceptions import ConfigError


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a YAML configuration file and return its contents as a dict.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Parsed YAML contents.

    Raises:
        ConfigError: If the file does not exist or cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML file {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    return data


def require_env(name: str) -> str:
    """Get a required environment variable or raise.

    Args:
        name: Environment variable name.

    Returns:
        The variable's value.

    Raises:
        ConfigError: If the variable is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in all values."
        )
    return value


def get_env(name: str, default: str) -> str:
    """Get an environment variable with an explicit default.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The variable's value or *default*.
    """
    return os.environ.get(name, default)


def load_env_file(path: Path | str | None = None) -> None:
    """Load a .env file into ``os.environ``.

    Existing variables are **not** overwritten.  When *path* is ``None`` the
    function walks up from the current working directory until it finds a
    ``.env`` file.

    Args:
        path: Explicit path to the ``.env`` file, or ``None`` to auto-detect.
    """
    if path is not None:
        env_path = Path(path)
    else:
        env_path = _find_env_file()

    if env_path is None or not env_path.exists():
        return

    _parse_env_file(env_path)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _find_env_file() -> Path | None:
    """Walk up from cwd looking for a .env file."""
    current = Path.cwd()
    while True:
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _parse_env_file(env_path: Path) -> None:
    """Parse KEY=VALUE lines from *env_path* into ``os.environ``."""
    with env_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
