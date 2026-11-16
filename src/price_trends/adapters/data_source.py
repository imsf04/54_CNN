from __future__ import annotations

import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from price_trends.references import MIRROR_DATA_URL, OFFICIAL_DATA_URL

DEFAULT_DATA_DIRS = (
    Path("data/raw/extracted"),
    Path("data/raw"),
    Path("/kaggle/input/price-trends"),
    Path("/content/price-trends"),
)


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    """Resolve a dataset directory across local, Kaggle, and Colab layouts."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_dir = os.getenv("PRICE_TRENDS_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(DEFAULT_DATA_DIRS)
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.dat")) and any(candidate.rglob("*.feather")):
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"未找到 .dat 和 .feather 数据文件；已检查: {searched}")


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"压缩包包含越界路径: {member.filename}")
        archive.extractall(destination)


def download_and_extract(
    output_dir: str | Path = "data/raw",
    url: str | None = None,
    timeout: int = 120,
) -> Path:
    """Download the cited archive and extract it into a reproducible directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    existing = next(
        (
            candidate
            for candidate in (destination, destination / "extracted")
            if candidate.exists()
            and any(candidate.rglob("*.dat"))
            and any(candidate.rglob("*.feather"))
        ),
        None,
    )
    if existing is not None:
        return existing
    archive_path = destination / "img_data.zip"
    source = url or OFFICIAL_DATA_URL
    request = Request(source, headers={"User-Agent": "price-trends-replication/1.0"})
    with urlopen(request, timeout=timeout) as response, archive_path.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    extracted = destination / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive_path, extracted)
    return extracted


__all__ = ["DEFAULT_DATA_DIRS", "MIRROR_DATA_URL", "download_and_extract", "resolve_data_dir"]
