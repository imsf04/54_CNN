from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from price_trends.references import PAPER_20D


class CNNBlock(nn.Sequential):
    """Paper building block: convolution, batch norm, leaky ReLU, max-pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        kernel = PAPER_20D.convolution_kernel
        padding = tuple(d * (k - 1) // 2 for k, d in zip(kernel, dilation))
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(PAPER_20D.leaky_relu_slope),
            nn.MaxPool2d(
                kernel_size=PAPER_20D.pooling_kernel,
                stride=PAPER_20D.pooling_kernel,
                ceil_mode=True,
            ),
        )


class PriceTrendCNN(nn.Module):
    """The paper-aligned 20-day CNN used in the course replication."""

    def __init__(self, filters: Iterable[int] | None = None, dropout: float | None = None) -> None:
        super().__init__()
        channels = tuple(filters or (64, 128, 256))
        if not channels:
            raise ValueError("至少需要一个卷积 block")
        blocks: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            blocks.append(
                CNNBlock(
                    in_channels,
                    out_channels,
                    stride=(PAPER_20D.first_vertical_stride, 1) if index == 0 else (1, 1),
                    dilation=(PAPER_20D.first_vertical_dilation, 1) if index == 0 else (1, 1),
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            feature_shape = self.features(torch.zeros(1, 1, PAPER_20D.image_height, PAPER_20D.image_width)).shape
        self.feature_shape = tuple(feature_shape[1:])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(PAPER_20D.dropout if dropout is None else dropout),
            nn.Linear(int(torch.tensor(self.feature_shape).prod()), 2),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

