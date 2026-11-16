"""Paper-aligned price-trend CNN replication package."""

from .data import DataBundle, ImageDataset, load_real_data, temporal_split
from .models import PriceTrendCNN

__all__ = ["DataBundle", "ImageDataset", "PriceTrendCNN", "load_real_data", "temporal_split"]
