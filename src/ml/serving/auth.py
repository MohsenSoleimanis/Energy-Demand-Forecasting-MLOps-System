"""API key authentication for the Energy Demand Forecasting API."""
import logging
import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_valid_api_keys() -> set[str]:
    """Load valid API keys from API_KEYS environment variable (comma-separated)."""
    raw = os.environ.get("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str:
    """FastAPI dependency that validates the X-API-Key header.

    Raises 401 if header is missing, 403 if key is invalid.
    Uses timing-safe comparison to prevent timing attacks.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    valid_keys = _get_valid_api_keys()
    if not valid_keys:
        logger.error("No API keys configured. Set API_KEYS in .env")
        raise HTTPException(status_code=500, detail="Server misconfiguration")

    if not any(secrets.compare_digest(api_key, k) for k in valid_keys):
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key
