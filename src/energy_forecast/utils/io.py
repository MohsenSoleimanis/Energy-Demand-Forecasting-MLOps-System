"""File I/O helpers for parquet, YAML, and directory management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd
import yaml

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to ensure exists.

    Returns
    -------
    Path
        The resolved :class:`Path` object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def read_parquet(path: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file into a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    path:
        Path to the ``.parquet`` file.
    **kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_parquet`.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_parquet(Path(path), **kwargs)


def write_parquet(df: pd.DataFrame, path: PathLike, **kwargs: Any) -> Path:
    """Write a :class:`~pandas.DataFrame` to a Parquet file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Destination file path.
    **kwargs:
        Extra keyword arguments forwarded to :meth:`pandas.DataFrame.to_parquet`.

    Returns
    -------
    Path
        The resolved path the file was written to.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False, **kwargs)
    return p


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def read_yaml(path: PathLike) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path:
        Path to the ``.yaml`` / ``.yml`` file.

    Returns
    -------
    dict
    """
    with open(Path(path), "r") as fh:
        return yaml.safe_load(fh) or {}


def write_yaml(data: dict[str, Any], path: PathLike) -> Path:
    """Write a dictionary to a YAML file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    data:
        Dictionary to serialise.
    path:
        Destination file path.

    Returns
    -------
    Path
        The resolved path the file was written to.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    return p
