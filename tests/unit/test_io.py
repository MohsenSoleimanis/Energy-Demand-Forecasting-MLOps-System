"""Tests for I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy_forecast.utils.io import (
    ensure_dir,
    read_parquet,
    read_yaml,
    write_parquet,
    write_yaml,
)

pytestmark = pytest.mark.unit


class TestIOHelpers:
    """Tests for I/O utility functions."""

    def test_parquet_roundtrip(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = tmp_path / "test.parquet"
        write_parquet(df, path)
        assert path.exists()

        loaded = read_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_yaml_roundtrip(self, tmp_path):
        data = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        path = tmp_path / "test.yaml"
        write_yaml(data, path)
        assert path.exists()

        loaded = read_yaml(path)
        assert loaded == data

    def test_ensure_dir(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        assert not new_dir.exists()
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
