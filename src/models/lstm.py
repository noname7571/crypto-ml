"""LSTM model for time-series price prediction (PyTorch)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """Stacked LSTM network for sequence regression or classification.

    Parameters
    ----------
    input_size:
        Number of input features per time step.
    hidden_size:
        Number of hidden units in each LSTM layer.
    num_layers:
        Number of stacked LSTM layers.
    output_size:
        Size of the final linear output (1 for regression).
    dropout:
        Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # take last time step
        return self.fc(out)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D feature array into overlapping sequences.

    Parameters
    ----------
    X:
        Shape ``(N, F)`` — N samples, F features.
    y:
        Shape ``(N,)`` — target values.
    seq_len:
        Number of time steps per sequence.

    Returns
    -------
    X_seq : np.ndarray, shape ``(N - seq_len, seq_len, F)``
    y_seq : np.ndarray, shape ``(N - seq_len,)``
    """
    X_seqs, y_seqs = [], []
    for i in range(seq_len, len(X)):
        X_seqs.append(X[i - seq_len : i])
        y_seqs.append(y[i])
    return np.array(X_seqs), np.array(y_seqs)


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap numpy arrays in PyTorch DataLoaders."""
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_lstm(
    model: LSTMModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train the LSTM model and return per-epoch loss history.

    Parameters
    ----------
    model:
        An instantiated :class:`LSTMModel`.
    train_loader / val_loader:
        PyTorch DataLoaders for training and validation.
    epochs:
        Number of training epochs.
    lr:
        Initial learning rate for Adam optimiser.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict with keys ``"train_loss"`` and ``"val_loss"``.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch).squeeze(-1)
                val_loss += criterion(pred, y_batch).item() * len(y_batch)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

    return history


def predict_lstm(
    model: LSTMModel,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Run inference and return predictions as a numpy array."""
    model.eval()
    model.to(device)
    tensor = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(tensor)
    loader = DataLoader(ds, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            preds.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(preds)
