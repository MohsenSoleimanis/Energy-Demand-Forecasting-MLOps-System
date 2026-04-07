"""Tests for LSTMForecaster."""

from __future__ import annotations

import numpy as np
import pytest

from energy_forecast.models.lstm_model import LSTMForecaster

pytestmark = pytest.mark.unit


class TestLSTMForecaster:
    """Tests for the LSTMForecaster class."""

    def test_create_sequences(self):
        model = LSTMForecaster(input_size=3, sequence_length=4, epochs=1)
        X = np.arange(30).reshape(10, 3).astype(np.float32)
        y = np.arange(10).astype(np.float32)
        X_seq, y_seq = model._create_sequences(X, y)
        # With length 10 and sequence_length 4, we get 10 - 4 = 6 sequences
        assert X_seq.shape == (6, 4, 3)
        assert y_seq.shape == (6,)
        # y_seq[0] should be y[4]
        assert y_seq[0] == y[4]

    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n_samples, n_features = 100, 3
        seq_len = 10
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, n_samples)

        model = LSTMForecaster(
            input_size=n_features,
            hidden_size=16,
            num_layers=1,
            sequence_length=seq_len,
            batch_size=16,
            epochs=3,
            learning_rate=0.01,
            patience=5,
            dropout=0.0,
        )
        metrics = model.fit(X, y)
        assert "train_loss" in metrics
        assert "epochs_trained" in metrics
        assert model.is_fitted is True

        # Predict on a single sequence
        X_pred = rng.standard_normal((seq_len, n_features))
        preds = model.predict(X_pred)
        assert len(preds) >= 1
        assert np.isfinite(preds).all()

    def test_save_and_load(self, tmp_path):
        rng = np.random.default_rng(42)
        n_samples, n_features = 50, 2
        seq_len = 5
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] + rng.normal(0, 0.1, n_samples)

        model = LSTMForecaster(
            input_size=n_features,
            hidden_size=8,
            num_layers=1,
            sequence_length=seq_len,
            batch_size=8,
            epochs=2,
            learning_rate=0.01,
            patience=5,
            dropout=0.0,
        )
        model.fit(X, y)

        save_dir = tmp_path / "lstm_save"
        model.save(save_dir)
        assert (save_dir / "lstm_model.pt").exists()

        model2 = LSTMForecaster(
            input_size=n_features,
            hidden_size=8,
            num_layers=1,
            sequence_length=seq_len,
            dropout=0.0,
        )
        model2.load(save_dir)
        assert model2.is_fitted is True

        X_test = rng.standard_normal((seq_len, n_features))
        preds_orig = model.predict(X_test)
        preds_loaded = model2.predict(X_test)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded, decimal=4)
