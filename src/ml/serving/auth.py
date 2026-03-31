"""API key authentication for the Energy Demand Forecasting API.

Reads the header name from serving config.  Uses timing-safe comparison
on every candidate key to prevent timing side-channels.
"""

import logging
import os
import secrets
from pathlib import Path

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.shared.config import load_config

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def _load_header_name() -> str:
    """Read the API key header name from serving config.

    Returns:
        Header name string (e.g. ``"X-API-Key"``).
    """
    try:
        cfg = load_config(_CONFIGS_DIR / "serving" / "api.yaml")
        return cfg.get("auth", {}).get("header_name", "X-API-Key")
    except Exception:
        return "X-API-Key"


API_KEY_HEADER = APIKeyHeader(name=_load_header_name(), auto_error=False)


def _get_valid_api_keys() -> set[str]:
    """Load valid API keys from the ``API_KEYS`` environment variable.

    Keys are expected as a comma-separated string.

    Returns:
        Set of non-empty key strings.
    """
    raw: str = os.environ.get("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str:
    """FastAPI dependency that validates the API key header.

    Uses :func:`secrets.compare_digest` on every configured key so that
    the total comparison time is constant regardless of which (or whether)
    a key matches.

    Args:
        api_key: Value extracted from the request header by FastAPI.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 401 if header is missing, 403 if invalid, 500 if
            no keys are configured on the server.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key header")

    valid_keys: set[str] = _get_valid_api_keys()
    if not valid_keys:
        logger.error("No API keys configured. Set API_KEYS env var.")
        raise HTTPException(status_code=500, detail="Server misconfiguration")

    # Evaluate ALL keys so elapsed time does not reveal which key matched.
    matched = False
    for candidate in valid_keys:
        if secrets.compare_digest(api_key, candidate):
            matched = True
            # Do NOT break — keep comparing remaining keys for constant time.

    if not matched:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key
