"""Load, validate, and split the project data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .adapters.data_source import resolve_data_dir
from .domain import DataBundle, SplitBundle
from .references import COURSE_DEV_END_YEAR, REPRODUCIBILITY_SEED


def make_synthetic_data(n: int = 32, seed: int = REPRODUCIBILITY_SEED) -> DataBundle:
    """Create a deterministic smoke-test dataset; never used for final results."""

    rng = np.random.default_rng(seed)
    images = rng.normal(0, 1, size=(n, 64, 60)).astype("float32")
    signal = images[:, :, :8].mean(axis=(1, 2))
    labels = (signal > np.median(signal)).astype("int64")
    years = np.resize(np.array([1993, 1994, 1995, 2000]), n)
    dates = np.array([f"{year}-01-15" for year in years], dtype="datetime64[D]")
    return DataBundle(images, labels, dates)


def _find_file(root: Path, suffix: str, horizon: str) -> Path:
    files = sorted(
        path
        for path in root.rglob(f"*{suffix}")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"未找到 {suffix} 文件: {root}")
    horizon_lower = horizon.lower().replace("-", "")
    matching = [path for path in files if horizon_lower in path.name.lower().replace("-", "")]
    return matching[0] if matching else files[0]


def _year_from_name(path: Path) -> int | None:
    match = re.search(r"_(\d{4})_(?:images|labels)", path.name)
    return int(match.group(1)) if match else None


def _target_column(frame, target: str) -> str:
    if "label" not in target.lower():
        preferred = [
            target.replace("Ret_", "Retx_") + "_label",
            target + "_label",
        ]
        for candidate in preferred:
            if candidate in frame.columns:
                return candidate
    if target in frame.columns:
        return target
    lower_target = target.lower()
    candidates = [column for column in frame.columns if lower_target in str(column).lower()]
    label_candidates = [column for column in candidates if "label" in str(column).lower()]
    if label_candidates:
        return label_candidates[0]
    if target.lower().startswith("retx_") and target.lower().endswith("_label"):
        raw_alias = target.replace("Retx_", "Ret_").replace("_label", "")
        if raw_alias in frame.columns:
            return raw_alias
    if candidates:
        return candidates[0]
    raise KeyError(f"找不到目标列 {target!r}; 可用列: {list(frame.columns)}")


def load_real_data(
    data_dir: str | Path | None = None,
    target: str = "Retx_20d_label",
    horizon: str = "20d",
    return_column: str | None = None,
) -> DataBundle:
    """Load and align one `.dat` image file with one Feather label file."""

    import pandas as pd

    root = resolve_data_dir(data_dir)
    image_files = {
        year: path
        for path in root.rglob("*.dat")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
        for year in [_year_from_name(path)]
        if year is not None and horizon.replace("-", "").lower() in path.name.lower().replace("-", "")
    }
    label_files = {
        year: path
        for path in root.rglob("*.feather")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
        for year in [_year_from_name(path)]
        if year is not None and horizon.replace("-", "").lower() in path.name.lower().replace("-", "")
    }
    if not image_files or not label_files:
        image_path = _find_file(root, ".dat", horizon)
        label_path = _find_file(root, ".feather", horizon)
        image_files, label_files = {0: image_path}, {0: label_path}
    years = sorted(set(image_files) & set(label_files))
    if not years:
        raise ValueError("没有找到可配对的年度图像和标签文件")
    image_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    stock_parts: list[np.ndarray] = []
    return_parts: list[np.ndarray] = []
    target_column = None
    for year in years:
        image_path, label_path = image_files[year], label_files[year]
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        if image_bytes.size % (64 * 60):
            raise ValueError(f"图像文件大小不能整除 64*60: {image_path}")
        images = image_bytes.reshape(-1, 64, 60)
        labels = pd.read_feather(label_path)
        if len(images) != len(labels):
            raise ValueError(f"图像数 {len(images)} 与标签行数 {len(labels)} 不一致: {image_path}, {label_path}")
        target_column = _target_column(labels, target)
        date_column = next((c for c in labels.columns if str(c).lower() == "date"), None)
        if date_column is None:
            raise KeyError("标签必须包含 Date/date 列")
        stock_column = next((c for c in labels.columns if str(c).lower() == "stockid"), None)
        if stock_column is None:
            raise KeyError("标签必须包含 StockID/stockid 列，无法建立严格 Date + StockID 配对")
        keys = labels[[date_column, stock_column]].copy()
        if keys.isna().any().any() or keys.duplicated().any():
            raise ValueError(f"Date + StockID 存在缺失或重复: {label_path}")
        y = pd.to_numeric(labels[target_column], errors="coerce")
        dates = pd.to_datetime(labels[date_column], errors="coerce")
        valid = y.isin([0, 1]) & dates.notna()
        valid_indices = np.flatnonzero(valid.to_numpy())
        image_parts.append(images[valid_indices])
        label_parts.append(y.loc[valid].to_numpy("int64"))
        date_parts.append(dates.loc[valid].to_numpy())
        stock_parts.append(labels.loc[valid, stock_column].to_numpy())
        if return_column and return_column in labels.columns:
            return_parts.append(pd.to_numeric(labels.loc[valid, return_column], errors="coerce").to_numpy("float32"))
    if not label_parts or sum(len(part) for part in label_parts) == 0:
        raise ValueError("删除缺失标签后没有有效二分类样本")
    returns = np.concatenate(return_parts) if return_parts and len(return_parts) == len(years) else None
    return DataBundle(
        images=np.concatenate(image_parts),
        labels=np.concatenate(label_parts),
        dates=np.concatenate(date_parts),
        stock_ids=np.concatenate(stock_parts),
        returns=returns,
        horizon=horizon,
    )


def temporal_split(bundle: DataBundle, dev_end: int = COURSE_DEV_END_YEAR) -> dict[str, DataBundle]:
    """Apply the course's 1993--1999 development / 2000+ test split."""

    years = np.asarray([int(str(date)[:4]) for date in bundle.dates])
    development = np.flatnonzero(years <= dev_end)
    test = np.flatnonzero(years > dev_end)
    if len(development) < 2 or len(test) == 0:
        raise ValueError("时间切分需要至少两个 development 样本和一个 test 样本")
    rng = np.random.default_rng(REPRODUCIBILITY_SEED)
    rng.shuffle(development)
    cut = max(1, int(len(development) * 0.7))
    if cut == len(development):
        cut -= 1

    def take(indices: np.ndarray) -> DataBundle:
        returns = None if bundle.returns is None else bundle.returns[indices]
        stock_ids = None if bundle.stock_ids is None else bundle.stock_ids[indices]
        return DataBundle(images=bundle.images[indices], labels=bundle.labels[indices], dates=bundle.dates[indices], stock_ids=stock_ids, returns=returns, horizon=bundle.horizon)

    return {"train": take(development[:cut]), "val": take(development[cut:]), "test": take(test)}


class ImageDataset(Dataset):
    def __init__(self, bundle: DataBundle) -> None:
        images = bundle.images.astype("float32")
        if images.size and images.max() > 1:
            images /= 255.0
        self.x = torch.from_numpy(images)[:, None]
        self.y = torch.from_numpy(bundle.labels.astype("int64"))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


__all__ = ["DataBundle", "ImageDataset", "SplitBundle", "load_real_data", "make_synthetic_data", "temporal_split"]
