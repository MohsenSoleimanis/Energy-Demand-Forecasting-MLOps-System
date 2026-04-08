"""Tests for TimeSeriesCV."""

from __future__ import annotations

import numpy as np
import pytest

from energy_forecast.training.cross_validation import TimeSeriesCV

pytestmark = pytest.mark.unit


class TestTimeSeriesCV:
    """Tests for the TimeSeriesCV class."""

    def test_expanding_window_splits(self):
        cv = TimeSeriesCV(n_splits=3, strategy="expanding_window", gap=0)
        X = np.arange(100).reshape(-1, 1)
        splits = list(cv.split(X))
        assert len(splits) > 0
        # Each successive fold should have a larger training set
        for i in range(1, len(splits)):
            assert len(splits[i][0]) >= len(splits[i - 1][0])

    def test_sliding_window_splits(self):
        cv = TimeSeriesCV(n_splits=3, strategy="sliding_window", gap=0)
        X = np.arange(200).reshape(-1, 1)
        splits = list(cv.split(X))
        assert len(splits) > 0
        # All training windows should have the same size (sliding)
        train_sizes = [len(s[0]) for s in splits]
        assert all(ts == train_sizes[0] for ts in train_sizes)

    def test_no_future_leakage(self):
        cv = TimeSeriesCV(n_splits=3, strategy="expanding_window", gap=5)
        X = np.arange(200).reshape(-1, 1)
        for train_idx, test_idx in cv.split(X):
            # The maximum training index + gap should be <= minimum test index
            assert train_idx.max() + cv.gap <= test_idx.min()

    def test_correct_number_of_splits(self):
        cv = TimeSeriesCV(n_splits=4, strategy="expanding_window", gap=0)
        X = np.arange(500).reshape(-1, 1)
        splits = list(cv.split(X))
        assert len(splits) == 4
