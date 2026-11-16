"""External-system adapters for local, Kaggle, and Colab execution."""

from .data_source import DEFAULT_DATA_DIRS, download_and_extract, resolve_data_dir
from .runtime import select_device

__all__ = ["DEFAULT_DATA_DIRS", "download_and_extract", "resolve_data_dir", "select_device"]
