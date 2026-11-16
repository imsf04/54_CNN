"""Control models used by the formal experiment runner."""

from __future__ import annotations

import torch
from torch import nn


class LogisticBaseline(nn.Module):
    def __init__(self, height: int = 64, width: int = 60) -> None:
        super().__init__()
        self.linear = nn.Linear(height * width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class MLPBaseline(nn.Module):
    def __init__(self, height: int = 64, width: int = 60) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
