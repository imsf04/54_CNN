from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from price_trends.data import (
    ImageDataset,
    image_pixel_stats,
    load_real_data,
    make_synthetic_data,
    temporal_split,
)
from price_trends.models import PriceTrendCNN
from price_trends.services.evaluation import classification_metrics, long_short_metrics
from price_trends.services.interpretation import grad_cam


def test_paper_20d_parameter_count_and_forward() -> None:
    model = PriceTrendCNN()
    assert model.feature_shape == (256, 3, 60)
    assert sum(parameter.numel() for parameter in model.parameters()) == 708866
    assert model(torch.zeros(2, 1, 64, 60)).shape == (2, 2)


def test_real_loader_preserves_image_label_alignment(tmp_path) -> None:
    images = np.concatenate([np.full((1, 64, 60), value, dtype=np.uint8) for value in [1, 2, 3, 4]])
    images.tofile(tmp_path / "images20d.dat")
    pd.DataFrame(
        {
            "Date": ["1993-01-01", "1993-01-02", "1993-01-03", "1993-01-04"],
            "StockID": ["A", "A", "B", "B"],
            "Ret_20d": [1, 2, 0, 1],
        }
    ).to_feather(tmp_path / "labels20d.feather")
    bundle = load_real_data(tmp_path)
    assert bundle.images[:, 0, 0].tolist() == [1, 3, 4]
    assert bundle.labels.tolist() == [1, 0, 1]


def test_lazy_loader_reads_aligned_rows_on_demand(tmp_path) -> None:
    images = np.concatenate([np.full((1, 64, 60), value, dtype=np.uint8) for value in [1, 2, 3, 4]])
    images.tofile(tmp_path / "images20d.dat")
    pd.DataFrame(
        {
            "Date": ["1993-01-01", "1993-01-02", "1993-01-03", "1993-01-04"],
            "StockID": ["A", "A", "B", "B"],
            "Ret_20d": [1, 2, 0, 1],
        }
    ).to_feather(tmp_path / "labels20d.feather")
    bundle = load_real_data(tmp_path, lazy=True)
    dataset = ImageDataset(bundle)
    assert dataset[0][0].shape == (1, 64, 60)
    assert np.isclose(float(dataset[0][0][0, 0, 0]), 1 / 255)
    assert dataset[0][1].item() == 1


def test_course_temporal_split_and_metrics() -> None:
    splits = temporal_split(make_synthetic_data(48))
    assert set(splits) == {"train", "val", "test"}
    assert all(int(str(date)[:4]) <= 1999 for date in splits["train"].dates)
    assert all(int(str(date)[:4]) > 1999 for date in splits["test"].dates)
    metrics = classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_grad_cam_returns_normalized_map() -> None:
    model = PriceTrendCNN()
    image = torch.rand(1, 1, 64, 60)
    heatmap = grad_cam(model, image)
    assert heatmap.ndim == 2
    assert float(heatmap.detach().min()) >= 0
    assert float(heatmap.detach().max()) <= 1


def test_image_stats_are_applied_after_byte_scaling() -> None:
    bundle = make_synthetic_data(4)
    dataset = ImageDataset(bundle)
    mean, std = image_pixel_stats(dataset, batch_size=2)
    normalized = ImageDataset(bundle, mean=mean, std=std)
    values = torch.stack([normalized[index][0] for index in range(len(normalized))])
    assert abs(float(values.mean())) < 1e-5
    assert abs(float(values.std(unbiased=False)) - 1.0) < 1e-5


def test_decile_portfolio_rejects_sparse_cross_sections() -> None:
    with pytest.raises(ValueError, match="too few assets"):
        long_short_metrics(
            np.array([0.1, 0.2, 0.8, 0.9]),
            np.array([0.01, 0.02, -0.01, 0.03]),
            np.array(["2000-01-31"] * 4),
            quantile=0.1,
        )
