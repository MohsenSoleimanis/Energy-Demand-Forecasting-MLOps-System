"""Simple file-backed feature store with versioning and metadata tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """File-backed feature store that persists feature sets as Parquet files.

    Each feature set is identified by a *name* and a *version* string.  Metadata
    (creation timestamp, shape, column names, arbitrary user config) is stored as
    a companion JSON sidecar next to the Parquet file.

    Directory layout::

        base_path/
          <name>/
            v<version>/
              features.parquet
              metadata.json

    Parameters
    ----------
    base_path:
        Root directory for all stored feature sets.  Created on first write
        if it does not exist.
    """

    FEATURES_FILENAME = "features.parquet"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, base_path: str = "data/features") -> None:
        self.base_path = Path(base_path)
        logger.info("feature_store_initialised", base_path=str(self.base_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_features(
        self,
        df: pd.DataFrame,
        name: str,
        version: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Persist a feature DataFrame and its metadata.

        Parameters
        ----------
        df:
            Feature DataFrame to store.
        name:
            Logical name of the feature set (e.g. ``"training_features"``).
        version:
            Version label (e.g. ``"1.0"`` or a timestamp string).
        metadata:
            Optional dict of extra metadata to record alongside the features.

        Returns
        -------
        Path
            Directory where the artefacts were written.
        """
        version_dir = self._version_dir(name, version)
        version_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = version_dir / self.FEATURES_FILENAME
        df.to_parquet(parquet_path, index=True, engine="pyarrow")

        meta: dict[str, Any] = {
            "name": name,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "feature_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        if metadata:
            meta["user_metadata"] = metadata

        meta_path = version_dir / self.METADATA_FILENAME
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

        logger.info(
            "features_saved",
            name=name,
            version=version,
            rows=len(df),
            cols=len(df.columns),
            path=str(version_dir),
        )
        return version_dir

    def load_features(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a feature set from the store.

        Parameters
        ----------
        name:
            Logical name of the feature set.
        version:
            Specific version to load.  If ``None`` the latest version
            (determined by directory modification time) is loaded.

        Returns
        -------
        pd.DataFrame
            The stored feature DataFrame.

        Raises
        ------
        FileNotFoundError
            If the requested feature set or version does not exist.
        """
        if version is None:
            version = self._resolve_latest_version(name)

        parquet_path = self._version_dir(name, version) / self.FEATURES_FILENAME
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Feature set '{name}' version '{version}' not found at {parquet_path}"
            )

        df = pd.read_parquet(parquet_path, engine="pyarrow")
        logger.info(
            "features_loaded",
            name=name,
            version=version,
            rows=len(df),
            cols=len(df.columns),
        )
        return df

    def list_versions(self, name: str) -> list[str]:
        """List all available versions for a given feature set name.

        Parameters
        ----------
        name:
            Logical name of the feature set.

        Returns
        -------
        list[str]
            Sorted list of version strings (ascending).
        """
        name_dir = self.base_path / name
        if not name_dir.exists():
            return []

        versions = []
        for child in sorted(name_dir.iterdir()):
            if child.is_dir() and (child / self.FEATURES_FILENAME).exists():
                # Strip the leading 'v' prefix used in directory names
                ver = child.name[1:] if child.name.startswith("v") else child.name
                versions.append(ver)

        return versions

    def get_metadata(self, name: str, version: Optional[str] = None) -> dict[str, Any]:
        """Retrieve stored metadata for a feature set version.

        Parameters
        ----------
        name:
            Logical name of the feature set.
        version:
            Specific version.  If ``None`` the latest version is used.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary.

        Raises
        ------
        FileNotFoundError
            If the metadata file does not exist.
        """
        if version is None:
            version = self._resolve_latest_version(name)

        meta_path = self._version_dir(name, version) / self.METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata for '{name}' version '{version}' not found at {meta_path}"
            )

        with open(meta_path) as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _version_dir(self, name: str, version: str) -> Path:
        """Return the directory path for a specific name/version pair."""
        return self.base_path / name / f"v{version}"

    def _resolve_latest_version(self, name: str) -> str:
        """Determine the latest version by directory modification time.

        Raises
        ------
        FileNotFoundError
            If no versions exist for the given name.
        """
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(f"No versions found for feature set '{name}'")

        # Pick the directory with the most recent modification time
        latest: str | None = None
        latest_mtime: float = -1.0
        for ver in versions:
            mtime = (self._version_dir(name, ver)).stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = ver

        assert latest is not None
        logger.info("resolved_latest_version", name=name, version=latest)
        return latest
