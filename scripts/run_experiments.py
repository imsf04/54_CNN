from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from price_trends.data import make_synthetic_data, temporal_split, ImageDataset
from price_trends.model import PriceTrendCNN
from price_trends.baselines import LogisticBaseline, MLPBaseline
from price_trends.train import fit
from torch.utils.data import DataLoader

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default='artifacts/experiments'); p.add_argument('--epochs', type=int, default=2); args = p.parse_args()
    results = {}
    for horizon in ('5d', '20d', '60d'):
        splits = temporal_split(make_synthetic_data(48, seed=7 + len(horizon))); train = DataLoader(ImageDataset(splits['train']), batch_size=8, shuffle=True); val = DataLoader(ImageDataset(splits['val']), batch_size=8)
        for name, model in [('logistic', LogisticBaseline()), ('mlp', MLPBaseline()), ('cnn', PriceTrendCNN())]:
            results[f'{horizon}_{name}'] = fit(model, train, val, epochs=args.epochs)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); (out / 'results.json').write_text(json.dumps(results, indent=2)); print(json.dumps(results, indent=2))
if __name__ == '__main__': main()
