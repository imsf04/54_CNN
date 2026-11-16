from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    cells = [
        md("""# Price Trends：Kaggle/Colab 正式 20-day 实验

按顺序运行所有代码单元。Notebook 使用同一份 20-day 图像，分别训练 Logistic、MLP 和 CNN，并保存 checkpoint、训练曲线、测试预测和指标。测试集只在最后评估。"""),
        code("""from pathlib import Path
import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = Path('/kaggle/working/54_CNN') if (Path('/kaggle/working/54_CNN') / 'src').exists() else Path('/content/54_CNN')
sys.path.insert(0, str(ROOT / 'src'))

from price_trends.adapters.runtime import select_device
from price_trends.data import ImageDataset, load_real_data, temporal_split
from price_trends.models import LogisticBaseline, MLPBaseline, PriceTrendCNN
from price_trends.references import PAPER_20D, REPRODUCIBILITY_SEED
from price_trends.services.evaluation import evaluate_model
from price_trends.services.training import fit, set_seed

set_seed(REPRODUCIBILITY_SEED)
DEVICE = select_device('auto')
DATA_DIR = Path(os.getenv('PRICE_TRENDS_DATA_DIR', '/kaggle/input/price-trends-20d'))
OUTPUT_DIR = ROOT / 'artifacts/kaggle_20d'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_EPOCHS = 100
print({'root': str(ROOT), 'data_dir': str(DATA_DIR), 'device': str(DEVICE), 'cuda': torch.cuda.is_available()})"""),
        md("""## 1. 数据审计与加载

确认数据目录包含 `.dat` 和 `.feather`。loader 会检查 64×60 图像、Date、StockID、标签有效性以及 Date + StockID 的缺失/重复；任何错误都会直接停止。"""),
        code("""from price_trends.data import load_real_data, temporal_split

bundle = load_real_data(DATA_DIR, target='Retx_20d_label', horizon='20d', return_column='Retx_20d')
splits = temporal_split(bundle)
display(pd.DataFrame({name: {'n': len(part.labels), 'positive_rate': part.labels.mean(), 'first': str(part.dates.min()), 'last': str(part.dates.max())} for name, part in splits.items()}).T)
assert set(splits) == {'train', 'val', 'test'}
assert all(int(str(date)[:4]) <= 1999 for date in splits['train'].dates)
assert all(int(str(date)[:4]) > 1999 for date in splits['test'].dates)"""),
        md("""## 2. 依次训练 Logistic、MLP、CNN

三种模型使用相同的样本和 DataLoader。每个模型单独保存最佳 checkpoint、训练历史和测试预测。"""),
        code("""loaders = {name: DataLoader(ImageDataset(part), batch_size=PAPER_20D.batch_size, shuffle=name == 'train') for name, part in splits.items()}
factories = {'logistic': LogisticBaseline, 'mlp': MLPBaseline, 'cnn': PriceTrendCNN}
models, histories, results = {}, {}, {}
for name, factory in factories.items():
    model = factory()
    history = fit(model, loaders['train'], loaders['val'], epochs=MAX_EPOCHS, device=DEVICE, checkpoint=OUTPUT_DIR / f'{name}_20d.pt')
    models[name], histories[name] = model, history
    results[name] = evaluate_model(model, splits['test'], loaders['test'], DEVICE)
    pd.DataFrame([results[name]]).to_json(OUTPUT_DIR / f'{name}_20d_metrics.json', orient='records', indent=2)
    pd.DataFrame([{'epoch': row['epoch'], 'train_loss': row['train']['loss'], 'val_loss': row['val']['loss'], 'train_accuracy': row['train']['accuracy'], 'val_accuracy': row['val']['accuracy']} for row in history]).to_csv(OUTPUT_DIR / f'{name}_20d_history.csv', index=False)
display(pd.DataFrame(results).T)"""),
        md("""## 3. 保存测试预测和训练曲线"""),
        code("""from price_trends.services.training import predict_proba
import matplotlib.pyplot as plt

prediction_frame = pd.DataFrame({'Date': splits['test'].dates, 'StockID': splits['test'].stock_ids, 'label': splits['test'].labels, 'return': splits['test'].returns})
for name, model in models.items():
    prediction_frame[f'{name}_probability'] = predict_proba(model, loaders['test'], DEVICE)
    history_frame = pd.read_csv(OUTPUT_DIR / f'{name}_20d_history.csv')
    history_frame.plot(x='epoch', y=['train_loss', 'val_loss'], marker='o', title=f'{name} loss')
    plt.show()
prediction_frame.to_parquet(OUTPUT_DIR / 'test_predictions.parquet', index=False)
pd.DataFrame(results).T.to_csv(OUTPUT_DIR / 'all_model_metrics.csv')
print(f'Saved artifacts to {OUTPUT_DIR}')"""),
        md("""## 4. 结果导出

下载 `artifacts/kaggle_20d/`，把 `all_model_metrics.csv`、`test_predictions.parquet`、训练曲线和 checkpoint 放入本地项目，再更新 `reports/replication_report.md`。不要把原始数据或 checkpoint 提交到 Git。"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    path = Path('notebooks/price_trends_replication.ipynb')
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


if __name__ == '__main__':
    main()
