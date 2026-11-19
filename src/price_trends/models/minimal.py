from __future__ import annotations

import torch
from torch import nn


class TemporalConv(nn.Module):
    """Convolve over the day axis only; the 64 price bins are input channels."""

    def __init__(self, filters: int = 16, kernel_size: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(64, filters, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.01),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters, 2)
        nn.init.xavier_uniform_(self.block[0].weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.block(images.squeeze(1))
        pooled = features.mean(dim=2)
        return self.classifier(self.dropout(pooled))
