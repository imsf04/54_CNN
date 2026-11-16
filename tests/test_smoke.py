import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import torch
from torch.utils.data import DataLoader
from price_trends.data import ImageDataset, make_synthetic_data, temporal_split
from price_trends.model import PriceTrendCNN
from price_trends.train import fit

def test_data_split_and_model_step(tmp_path):
    splits = temporal_split(make_synthetic_data(24))
    assert set(splits) == {"train", "val", "test"}
    model = PriceTrendCNN()
    history = fit(model, DataLoader(ImageDataset(splits["train"]), batch_size=4), DataLoader(ImageDataset(splits["val"]), batch_size=4), epochs=1, checkpoint=tmp_path / "m.pt")
    assert history and (tmp_path / "m.pt").exists()
    assert model(torch.randn(2, 1, 64, 60)).shape == (2, 2)
