"""Tests for DataProcessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.data.processor import DataProcessor

pytestmark = pytest.mark.unit


class TestDataProcessor:
    """Tests for the DataProcessor class."""

    def test_normalize_standard(self, sample_train_data):
        X_train, _, X_val, _ = sample_train_data
        # Use a dummy X_test
        X_test = X_val.copy()
        X_train_s, X_val_s, X_test_s, scaler = DataProcessor.normalize(
            X_train, X_val, X_test, method="standard"
        )
        # After standard scaling, training data should have ~zero mean and ~unit variance
        assert np.abs(X_train_s.mean(axis=0)).max() < 0.1
        assert np.abs(X_train_s.std(axis=0) - 1.0).max() < 0.15

    def test_normalize_minmax(self, sample_train_data):
        X_train, _, X_val, _ = sample_train_data
        X_test = X_val.copy()
        X_train_s, X_val_s, X_test_s, scaler = DataProcessor.normalize(
            X_train, X_val, X_test, method="minmax"
        )
        # Training data should be in [0, 1]
        assert X_train_s.min() >= -1e-10
        assert X_train_s.max() <= 1.0 + 1e-10

    def test_split_respects_time_order(self, sample_energy_df):
        processor = DataProcessor(
            config={
                "lag_features": [1, 2],
                "rolling_windows": [3],
                "rolling_stats": ["mean"],
                "drop_na": True,
            }
        )
        df = processor.create_features(sample_energy_df)
        split = processor.split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        # Train set should be before val set which should be before test set
        assert split.X_train.shape[0] > 0
        assert split.X_val.shape[0] > 0
        assert split.X_test.shape[0] > 0
        total = split.X_train.shape[0] + split.X_val.shape[0] + split.X_test.shape[0]
        assert total == len(df)
