"""Backward-compatible data download imports."""

from .adapters.data_source import download_and_extract

__all__ = ["download_and_extract"]
