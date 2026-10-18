"""LSTM forecaster with mini-batch training."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMForecaster(nn.Module):
    """Same network as the notebook: one LSTM layer and a linear head on the last step."""

    def __init__(self, input_size: int = 6, hidden_size: int = 100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_lstm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_size: int = 100,
    seed: int = 0,
    device: torch.device | None = None,
) -> LSTMForecaster:
    """Fit on windows x (n, seq_len, features) and targets y (n,)."""
    set_seed(seed)
    device = device or pick_device()
    model = LSTMForecaster(x.shape[2], hidden_size).to(device)
    data = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: LSTMForecaster, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
