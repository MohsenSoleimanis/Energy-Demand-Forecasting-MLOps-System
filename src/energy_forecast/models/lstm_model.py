"""LSTM-based energy demand forecaster using PyTorch."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional

from energy_forecast.models.base import BaseForecaster


class LSTMNetwork(nn.Module):
    """LSTM neural network for time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecaster implementing the BaseForecaster interface."""

    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, sequence_length: int = 168, batch_size: int = 64,
                 epochs: int = 50, learning_rate: float = 0.001, patience: int = 10, **kwargs):
        super().__init__(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, sequence_length=sequence_length, batch_size=batch_size,
            epochs=epochs, learning_rate=learning_rate, patience=patience, **kwargs
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMNetwork] = None
        self.training_losses: list[float] = []
        self.val_losses: list[float] = []

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """Train the LSTM model."""
        # Reshape X if 1D
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_val is not None and X_val.ndim == 1:
            X_val = X_val.reshape(-1, 1)

        self.input_size = X_train.shape[1]

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_train, y_train)

        # Create DataLoader
        dataset = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq))
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        self.model = LSTMNetwork(self.input_size, self.hidden_size, self.num_layers, self.dropout_rate).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        self.training_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.training_losses.append(avg_train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        predictions = self.model(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()
                        n_val_batches += 1
                avg_val_loss = val_loss / max(n_val_batches, 1)
                self.val_losses.append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state
                    self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        # Restore best model
                        if hasattr(self, "_best_state"):
                            self.model.load_state_dict(self._best_state)
                        break

        self.is_fitted = True
        metrics = {
            "train_loss": self.training_losses[-1],
            "epochs_trained": len(self.training_losses),
            "train_rmse": float(np.sqrt(self.training_losses[-1])),
        }
        if self.val_losses:
            metrics["val_loss"] = self.val_losses[-1]
            metrics["val_rmse"] = float(np.sqrt(self.val_losses[-1]))
            metrics["best_val_loss"] = best_val_loss
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained LSTM."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model.eval()
        # For prediction, if X has enough rows, create sequences
        # If X is already shaped as sequences (3D), use directly
        if X.ndim == 2:
            if len(X) <= self.sequence_length:
                # Treat as single sequence
                X_seq = X[np.newaxis, :, :].astype(np.float32)
            else:
                X_seq = np.array([X[i:i + self.sequence_length]
                                  for i in range(len(X) - self.sequence_length + 1)], dtype=np.float32)
        else:
            X_seq = X.astype(np.float32)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "params": self.params,
                "input_size": self.input_size,
                "training_losses": self.training_losses,
                "val_losses": self.val_losses,
            }, path / "lstm_model.pt")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        checkpoint = torch.load(path / "lstm_model.pt", map_location=self.device, weights_only=False)
        self.input_size = checkpoint["input_size"]
        self.model = LSTMNetwork(self.input_size, self.hidden_size, self.num_layers, self.dropout_rate).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_losses = checkpoint.get("training_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.is_fitted = True
