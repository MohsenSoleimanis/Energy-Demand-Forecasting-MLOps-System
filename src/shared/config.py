"""Centralized configuration. All credentials come from environment variables."""
import logging
import os

logger = logging.getLogger(__name__)

def require_env(name: str) -> str:
    """Get a required environment variable or raise with clear message."""
    val = os.environ.get(name)
    if not val:
        raise OSError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in all values."
        )
    return val

def get_s3_config() -> dict:
    """Return S3/MinIO configuration from environment variables."""
    return {
        "endpoint_url": os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"),
        "aws_access_key_id": require_env("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": require_env("AWS_SECRET_ACCESS_KEY"),
    }

def get_s3_client():
    """Create a boto3 S3 client using environment credentials."""
    import boto3
    return boto3.client("s3", **get_s3_config())

def load_env_file(env_path: str | None = None):
    """Load .env file into os.environ. Does NOT override existing vars."""
    from pathlib import Path
    if env_path is None:
        # Search up from cwd for .env
        path = Path.cwd()
        while path != path.parent:
            candidate = path / ".env"
            if candidate.exists():
                env_path = str(candidate)
                break
            path = path.parent
    if env_path is None:
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
