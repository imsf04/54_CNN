"""Load, validate, and split the project data."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .adapters.data_source import resolve_data_dir
from .domain import DataBundle, LazyDataBundle, SplitBundle
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


def _discover_pairs(root: Path, horizon: str) -> tuple[dict[int, Path], dict[int, Path], list[int]]:
    horizon_key = horizon.replace("-", "").lower()
    all_images = sorted(path for path in root.rglob("*.dat") if "__MACOSX" not in path.parts and not path.name.startswith("._"))
    all_labels = sorted(path for path in root.rglob("*.feather") if "__MACOSX" not in path.parts and not path.name.startswith("._"))
    image_files = {
        year: path
        for path in root.rglob("*.dat")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
        for year in [_year_from_name(path)]
        if year is not None and horizon_key in path.name.lower().replace("-", "")
    }
    label_files = {
        year: path
        for path in root.rglob("*.feather")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
        for year in [_year_from_name(path)]
        if year is not None and horizon_key in path.name.lower().replace("-", "")
    }
    if not image_files or not label_files:
        has_year_named_files = any(_year_from_name(path) is not None for path in all_images + all_labels)
        if has_year_named_files or len(all_images) != 1 or len(all_labels) != 1:
            raise FileNotFoundError(f"未找到 horizon={horizon!r} 的成对图像和标签文件: {root}")
        image_files, label_files = {0: all_images[0]}, {0: all_labels[0]}
    years = sorted(set(image_files) & set(label_files))
    if not years:
        raise ValueError("没有找到可配对的年度图像和标签文件")
    return image_files, label_files, years


def load_real_data(
    data_dir: str | Path | None = None,
    target: str = "Retx_20d_label",
    horizon: str = "20d",
    return_column: str | None = None,
    max_samples: int | None = None,
    lazy: bool = False,
) -> DataBundle | LazyDataBundle:
    """Load aligned data, optionally keeping image bytes on disk until batching."""

    import pandas as pd

    root = resolve_data_dir(data_dir)
    image_files, label_files, years = _discover_pairs(root, horizon)

    def load_year(
        year: int,
    ) -> tuple[
        int,
        Path,
        int,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        str,
    ]:
        image_path, label_path = image_files[year], label_files[year]
        image_bytes = np.memmap(image_path, dtype=np.uint8, mode="r")
        if image_bytes.size % (64 * 60):
            raise ValueError(f"图像文件大小不能整除 64*60: {image_path}")
        image_count = image_bytes.size // (64 * 60)
        images = image_bytes.reshape(image_count, 64, 60)
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
        if max_samples is not None:
            per_year = max(1, max_samples // len(years))
            rng = np.random.default_rng(7 + year)
            valid_indices = np.sort(
                rng.choice(valid_indices, size=min(per_year, len(valid_indices)), replace=False)
            )
        selected_returns = None
        if return_column and return_column in labels.columns:
            selected_returns = pd.to_numeric(
                labels.iloc[valid_indices][return_column], errors="coerce"
            ).to_numpy("float32")
        selected_images = None if lazy else np.asarray(images[valid_indices])
        return (
            year,
            image_path,
            image_count,
            selected_images,
            valid_indices.astype("int32"),
            y.iloc[valid_indices].to_numpy("int64"),
            dates.iloc[valid_indices].to_numpy(),
            labels.iloc[valid_indices][stock_column].to_numpy(),
            selected_returns,
            target_column,
        )

    if lazy or (max_samples is not None and len(years) > 1):
        workers = min(4, len(years), max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            year_parts = list(executor.map(load_year, years))
    else:
        year_parts = [load_year(year) for year in years]

    target_columns = {part[-1] for part in year_parts}
    if len(target_columns) != 1:
        raise ValueError(f"年度标签文件的目标列不一致: {sorted(target_columns)}")
    label_parts = [part[5] for part in year_parts]
    date_parts = [part[6] for part in year_parts]
    stock_parts = [part[7] for part in year_parts]
    return_parts = [part[8] for part in year_parts if part[8] is not None]
    if not label_parts or sum(len(part) for part in label_parts) == 0:
        raise ValueError("删除缺失标签后没有有效二分类样本")
    returns = np.concatenate(return_parts) if len(return_parts) == len(years) else None
    labels = np.concatenate(label_parts)
    dates = np.concatenate(date_parts)
    stock_ids = np.concatenate(stock_parts)
    if lazy:
        image_paths = tuple(part[1] for part in year_parts)
        image_counts = tuple(part[2] for part in year_parts)
        file_ids = np.concatenate(
            [np.full(len(part[4]), index, dtype="int16") for index, part in enumerate(year_parts)]
        )
        row_indices = np.concatenate([part[4] for part in year_parts])
        return LazyDataBundle(
            image_paths=image_paths,
            image_counts=image_counts,
            file_ids=file_ids,
            row_indices=row_indices,
            labels=labels,
            dates=dates,
            stock_ids=stock_ids,
            returns=returns,
            horizon=horizon,
        )
    image_parts = [part[3] for part in year_parts if part[3] is not None]
    return DataBundle(
        images=np.concatenate(image_parts),
        labels=labels,
        dates=dates,
        stock_ids=stock_ids,
        returns=returns,
        horizon=horizon,
    )


def temporal_split(
    bundle: DataBundle | LazyDataBundle,
    dev_end: int = COURSE_DEV_END_YEAR,
) -> dict[str, DataBundle | LazyDataBundle]:
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

    def take(indices: np.ndarray) -> DataBundle | LazyDataBundle:
        returns = None if bundle.returns is None else bundle.returns[indices]
        stock_ids = None if bundle.stock_ids is None else bundle.stock_ids[indices]
        if isinstance(bundle, LazyDataBundle):
            return LazyDataBundle(
                image_paths=bundle.image_paths,
                image_counts=bundle.image_counts,
                file_ids=bundle.file_ids[indices],
                row_indices=bundle.row_indices[indices],
                labels=bundle.labels[indices],
                dates=bundle.dates[indices],
                stock_ids=stock_ids,
                returns=returns,
                horizon=bundle.horizon,
            )
        return DataBundle(images=bundle.images[indices], labels=bundle.labels[indices], dates=bundle.dates[indices], stock_ids=stock_ids, returns=returns, horizon=bundle.horizon)

    return {"train": take(development[:cut]), "val": take(development[cut:]), "test": take(test)}


def image_pixel_stats(dataset: Dataset, batch_size: int = 128) -> tuple[float, float]:
    """Compute mean and population std from already scaled training images only."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    total_sum = 0.0
    total_squared_sum = 0.0
    for images, _ in loader:
        values = images.to(dtype=torch.float64)
        total += values.numel()
        total_sum += float(values.sum())
        total_squared_sum += float((values * values).sum())
    if total == 0:
        raise ValueError("cannot compute image statistics from an empty dataset")
    mean = total_sum / total
    variance = total_squared_sum / total - mean * mean
    if variance < 0:
        raise ValueError(f"training image variance is negative: {variance}")
    std = variance**0.5
    if std <= 0:
        raise ValueError("training images have zero pixel variance")
    return mean, std


class ImageDataset(Dataset):
    def __init__(
        self,
        bundle: DataBundle | LazyDataBundle,
        *,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("image normalization requires finite mean and positive std")
        self.y = torch.from_numpy(bundle.labels.astype("int64"))
        self.mean = float(mean)
        self.std = float(std)
        self.x = None
        self._maps: dict[int, np.memmap] = {}
        self._bundle = bundle
        if isinstance(bundle, DataBundle):
            images = bundle.images.astype("float32")
            if images.size and images.max() > 1:
                images /= 255.0
            self.x = (torch.from_numpy(images)[:, None] - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        if self.x is not None:
            return self.x[index], self.y[index]
        bundle = self._bundle
        file_id = int(bundle.file_ids[index])
        image_map = self._maps.get(file_id)
        if image_map is None:
            image_map = np.memmap(
                bundle.image_paths[file_id],
                dtype=np.uint8,
                mode="r",
                shape=(bundle.image_counts[file_id], 64, 60),
            )
            self._maps[file_id] = image_map
        image = np.asarray(image_map[int(bundle.row_indices[index])], dtype="float32") / 255.0
        return (torch.from_numpy(image[None]) - self.mean) / self.std, self.y[index]


class IterableImageDataset(IterableDataset):
    """Yield normalized image batches straight from the per-year memmaps."""

    def __init__(
        self,
        bundle: LazyDataBundle,
        *,
        batch_size: int,
        mean: float,
        std: float,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("image normalization requires finite mean and positive std")
        self.bundle = bundle
        self.batch_size = batch_size
        self.mean = float(mean)
        self.std = float(std)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return -(-len(self.bundle.labels) // self.batch_size)

    def _image_map(self, file_id: int) -> np.memmap:
        if file_id not in self._maps:
            self._maps[file_id] = np.memmap(
                self.bundle.image_paths[file_id],
                dtype=np.uint8,
                mode="r",
                shape=(self.bundle.image_counts[file_id], 64, 60),
            )
        return self._maps[file_id]

    def __iter__(self):
        total = len(self.bundle.labels)
        order = self.rng.permutation(total) if self.shuffle else np.arange(total)
        for start in range(0, total, self.batch_size):
            indices = order[start : start + self.batch_size]
            file_ids = self.bundle.file_ids[indices]
            rows = self.bundle.row_indices[indices]
            images = np.empty((len(indices), 64, 60), dtype="float32")
            for file_id in np.unique(file_ids):
                selected = file_ids == file_id
                images[selected] = self._image_map(int(file_id))[rows[selected]]
            images /= 255.0
            x = (torch.from_numpy(images).unsqueeze(1) - self.mean) / self.std
            y = torch.from_numpy(self.bundle.labels[indices].astype("int64"))
            yield x, y


__all__ = [
    "DataBundle",
    "ImageDataset",
    "IterableImageDataset",
    "LazyDataBundle",
    "SplitBundle",
    "image_pixel_stats",
    "load_real_data",
    "make_synthetic_data",
    "temporal_split",
]
