from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DataBundle:
    """Aligned image observations and their target metadata."""

    images: np.ndarray
    labels: np.ndarray
    dates: np.ndarray
    stock_ids: np.ndarray | None = None
    returns: Optional[np.ndarray] = None
    horizon: str = "20d"

    def __post_init__(self) -> None:
        n = len(self.labels)
        if len(self.images) != n or len(self.dates) != n:
            raise ValueError("images, labels, and dates must have the same length")
        if self.stock_ids is not None and len(self.stock_ids) != n:
            raise ValueError("stock_ids must have the same length as labels")
        if self.returns is not None and len(self.returns) != n:
            raise ValueError("returns must have the same length as labels")
        if self.images.ndim != 3 or self.images.shape[1:] != (64, 60):
            raise ValueError("images must have shape (N, 64, 60)")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("labels must contain only 0 and 1 after filtering")


@dataclass(frozen=True)
class LazyDataBundle:
    """Metadata and row references for out-of-core image loading."""

    image_paths: tuple[Path, ...]
    image_counts: tuple[int, ...]
    file_ids: np.ndarray
    row_indices: np.ndarray
    labels: np.ndarray
    dates: np.ndarray
    stock_ids: np.ndarray | None = None
    returns: Optional[np.ndarray] = None
    horizon: str = "20d"

    def __post_init__(self) -> None:
        n = len(self.labels)
        if not self.image_paths or len(self.image_paths) != len(self.image_counts):
            raise ValueError("image_paths and image_counts must describe every image file")
        if len(self.file_ids) != n or len(self.row_indices) != n or len(self.dates) != n:
            raise ValueError("lazy image references and metadata must have the same length")
        if self.stock_ids is not None and len(self.stock_ids) != n:
            raise ValueError("stock_ids must have the same length as labels")
        if self.returns is not None and len(self.returns) != n:
            raise ValueError("returns must have the same length as labels")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("labels must contain only 0 and 1 after filtering")
        if np.any(self.file_ids < 0) or np.any(self.file_ids >= len(self.image_paths)):
            raise ValueError("file_ids contain an unknown image file")
        counts = np.asarray(self.image_counts)
        if np.any(self.row_indices < 0) or np.any(self.row_indices >= counts[self.file_ids]):
            raise ValueError("row_indices contain an invalid image row")


@dataclass(frozen=True)
class SplitBundle:
    """The development train/validation split and untouched out-of-sample test."""

    train: DataBundle | LazyDataBundle
    val: DataBundle | LazyDataBundle
    test: DataBundle | LazyDataBundle

    def as_dict(self) -> dict[str, DataBundle | LazyDataBundle]:
        return {"train": self.train, "val": self.val, "test": self.test}
