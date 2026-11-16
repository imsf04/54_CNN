from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataBundle:
    images: np.ndarray
    labels: np.ndarray
    dates: np.ndarray


def make_synthetic_data(n: int = 32, seed: int = 7) -> DataBundle:
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 1, size=(n, 64, 60)).astype("float32")
    signal = images[:, :, :8].mean(axis=(1, 2))
    labels = (signal > np.median(signal)).astype("int64")
    years = np.resize(np.array([1993, 1994, 1995, 2000]), n)
    dates = np.array([f"{y}-01-15" for y in years], dtype="datetime64[D]")
    return DataBundle(images, labels, dates)


def _target_column(frame: pd.DataFrame, target: str) -> str:
    if target in frame.columns:
        return target
    candidates = [c for c in frame.columns if target.lower() in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError(f"Target {target!r} not found in labels: {list(frame.columns)}")


def load_real_data(data_dir: str | Path, target: str = "Ret_20d") -> DataBundle:
    """Load a matching image `.dat` and Feather label file from a data directory."""
    import pandas as pd
    root = Path(data_dir)
    image_files = sorted(root.glob("*.dat"))
    label_files = sorted(root.glob("*.feather"))
    if not image_files or not label_files:
        raise FileNotFoundError("Expected at least one .dat image file and one .feather label file")
    image_path = next((p for p in image_files if "20d" in p.name.lower()), image_files[0])
    label_path = next((p for p in label_files if "20d" in p.name.lower()), label_files[0])
    images = np.fromfile(image_path, dtype=np.uint8)
    if images.size % (64 * 60):
        raise ValueError(f"Image file size is not divisible by 64*60: {image_path}")
    images = images.reshape(-1, 64, 60)
    labels = pd.read_feather(label_path)
    target_col = _target_column(labels, target)
    date_col = next((c for c in labels.columns if c.lower() == "date"), None)
    if date_col is None:
        raise KeyError("Labels must contain a Date column")
    y = pd.to_numeric(labels[target_col], errors="coerce")
    dates = pd.to_datetime(labels[date_col], errors="coerce")
    valid = y.isin([0, 1]) & dates.notna()
    count = min(len(images), int(valid.sum()))
    if count == 0:
        raise ValueError("No valid binary labels remain after removing missing values")
    return DataBundle(images[:count], y[valid].to_numpy("int64")[:count], dates[valid].reset_index(drop=True).to_numpy()[:count])


def temporal_split(bundle: DataBundle, dev_end: int = 1999) -> dict[str, DataBundle]:
    years = np.array([int(str(d)[:4]) for d in bundle.dates])
    dev = years <= dev_end
    test = ~dev
    indices = np.arange(len(bundle.labels))
    rng = np.random.default_rng(7)
    shuffled = indices[dev].copy()
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.7))

    def take(idx: np.ndarray) -> DataBundle:
        return DataBundle(bundle.images[idx], bundle.labels[idx], bundle.dates[idx])

    return {"train": take(shuffled[:cut]), "val": take(shuffled[cut:]), "test": take(indices[test])}


class ImageDataset(Dataset):
    def __init__(self, bundle: DataBundle):
        self.x = torch.from_numpy(bundle.images.astype("float32"))[:, None]
        self.y = torch.from_numpy(bundle.labels.astype("int64"))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
