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

按顺序运行所有代码单元。Notebook 使用同一份 20-day 图像，分别训练 Logistic、MLP 和 CNN，并保存 checkpoint、训练曲线、测试预测和指标。测试集只在最后评估。首次运行会把 Kaggle 只读输入盘上的数据 staging 到 `/kaggle/working` 本地盘，避免随机读取远端挂载。"""),
        code("""from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import subprocess
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

WORKSPACE = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
ROOT = WORKSPACE / '54_CNN'
REPO_URL = 'https://github.com/imsf04/54_CNN.git'
REPO_REF = 'stevenPeng/kaggle-rigorous-replication'
if (ROOT / '.git').exists():
    subprocess.run(['git', '-C', str(ROOT), 'fetch', 'origin', REPO_REF], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'reset', '--hard', f'origin/{REPO_REF}'], check=True)
elif ROOT.exists():
    shutil.rmtree(ROOT)
    subprocess.run(['git', 'clone', REPO_URL, str(ROOT)], check=True)
else:
    subprocess.run(['git', 'clone', REPO_URL, str(ROOT)], check=True)
subprocess.run(['git', '-C', str(ROOT), 'fetch', 'origin', REPO_REF], check=True)
subprocess.run(['git', '-C', str(ROOT), 'reset', '--hard', f'origin/{REPO_REF}'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(ROOT)], check=True)
sys.path.insert(0, str(ROOT / 'src'))

from price_trends.adapters.runtime import select_device
from price_trends.data import ImageDataset, load_real_data, temporal_split
from price_trends.models import LogisticBaseline, MLPBaseline, PriceTrendCNN
from price_trends.references import PAPER_20D, REPRODUCIBILITY_SEED
from price_trends.services.evaluation import evaluate_model
from price_trends.services.training import fit, set_seed

set_seed(REPRODUCIBILITY_SEED)
DEVICE = select_device('auto')
CPU_THREADS = min(4, max(1, os.cpu_count() or 1))
torch.set_num_threads(CPU_THREADS)
if DEVICE.type == 'cuda':
    for device_id in range(torch.cuda.device_count()):
        torch.empty(1, device=f'cuda:{device_id}')
    torch.cuda.set_device(0)
DEFAULT_DATA_INPUT = '/kaggle/input/price-trends-20d' if Path('/kaggle/input').exists() else '/content/price-trends-20d'
DATA_INPUT = Path(os.getenv('PRICE_TRENDS_DATA_DIR', DEFAULT_DATA_INPUT))
if not DATA_INPUT.exists() and Path('/kaggle/input').exists():
    DATA_INPUT = Path('/kaggle/input/datasets/imsfp04/price-trends-20d')
OUTPUT_DIR = ROOT / 'artifacts/kaggle_20d'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_EPOCHS = 100
MAX_SAMPLES_ENV = os.getenv('PRICE_TRENDS_MAX_SAMPLES')
MAX_SAMPLES = int(MAX_SAMPLES_ENV) if MAX_SAMPLES_ENV else None
LAZY_IMAGES = True
archives = sorted(DATA_INPUT.rglob('*.zip'))
if archives:
    import zipfile
    DATA_DIR = ROOT / 'data/kaggle_extracted'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = archives[0]
    if not any(DATA_DIR.rglob('*.dat')):
        print(f'Extracting {archive} ...')
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(DATA_DIR)
        print(f'Extracted to {DATA_DIR}')
else:
    DATA_DIR = DATA_INPUT

def has_data(root):
    return any(root.rglob('*.dat')) and any(root.rglob('*.feather'))

if str(DATA_DIR).startswith('/kaggle/input'):
    staged_dir = ROOT / 'data/kaggle_staged'
    source_files = sorted([*DATA_DIR.rglob('*.dat'), *DATA_DIR.rglob('*.feather')])
    pending = [
        source for source in source_files
        if not (staged_dir / source.relative_to(DATA_DIR)).exists()
        or (staged_dir / source.relative_to(DATA_DIR)).stat().st_size != source.stat().st_size
    ]
    if pending:
        print(f'Staging {len(pending)} data files onto local working disk ...')

        def copy_one(source):
            target = staged_dir / source.relative_to(DATA_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return target

        with ThreadPoolExecutor(max_workers=2) as executor:
            for index, target in enumerate(executor.map(copy_one, pending), start=1):
                print(f'[{index}/{len(pending)}] staged {target.name}')
    if not has_data(staged_dir):
        raise FileNotFoundError(f'本地 staging 后仍未找到 .dat/.feather: {staged_dir}')
    DATA_DIR = staged_dir
print({'root': str(ROOT), 'revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD']).decode().strip(), 'data_dir': str(DATA_DIR), 'device': str(DEVICE), 'cuda': torch.cuda.is_available(), 'cuda_devices': torch.cuda.device_count(), 'cpu_threads': CPU_THREADS, 'lazy_images': LAZY_IMAGES, 'max_samples': MAX_SAMPLES})"""),
        md("""## 1. 数据审计与加载

确认数据目录包含 `.dat` 和 `.feather`。loader 会检查 64×60 图像、Date、StockID、标签有效性以及 Date + StockID 的缺失/重复；任何错误都会直接停止。"""),
        code("""from price_trends.data import ImageDataset, image_pixel_stats, load_real_data, temporal_split

sample_message = f'{MAX_SAMPLES:,}' if MAX_SAMPLES is not None else 'all available'
print(f'Loading {sample_message} samples with lazy memmap batches ...')
bundle = load_real_data(DATA_DIR, target='Retx_20d_label', horizon='20d', return_column='Retx_20d', max_samples=MAX_SAMPLES, lazy=LAZY_IMAGES)
print(f'Loaded metadata for {len(bundle.labels):,} samples; lazy_images={LAZY_IMAGES}')
splits = temporal_split(bundle)
display(pd.DataFrame({name: {'n': len(part.labels), 'positive_rate': part.labels.mean(), 'first': str(part.dates.min()), 'last': str(part.dates.max())} for name, part in splits.items()}).T)
assert set(splits) == {'train', 'val', 'test'}
assert all(int(str(date)[:4]) <= 1999 for date in splits['train'].dates)
assert all(int(str(date)[:4]) > 1999 for date in splits['test'].dates)"""),
        md("""## 2. 依次训练 Logistic、MLP、CNN

三种模型使用相同的样本和 DataLoader。检测到多张 CUDA 卡时，Notebook 使用双 GPU 数据并行；每个模型单独保存最佳 checkpoint、训练历史和测试预测。"""),
        code("""NUM_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
LOADER_OPTIONS = {'batch_size': PAPER_20D.batch_size, 'pin_memory': DEVICE.type == 'cuda'}
if NUM_WORKERS:
    LOADER_OPTIONS.update({'num_workers': NUM_WORKERS, 'persistent_workers': True, 'prefetch_factor': 2})
normalization_mean, normalization_std = image_pixel_stats(ImageDataset(splits['train']), batch_size=PAPER_20D.batch_size)
datasets = {name: ImageDataset(part, mean=normalization_mean, std=normalization_std) for name, part in splits.items()}
print({'normalization_mean': normalization_mean, 'normalization_std': normalization_std, 'source': 'train split only'})
loaders = {name: DataLoader(dataset, shuffle=name == 'train', **LOADER_OPTIONS) for name, dataset in datasets.items()}
factories = {'logistic': LogisticBaseline, 'mlp': MLPBaseline, 'cnn': PriceTrendCNN}
MULTI_GPU = DEVICE.type == 'cuda' and torch.cuda.device_count() > 1
GPU_IDS = list(range(torch.cuda.device_count())) if MULTI_GPU else []
print({'multi_gpu': MULTI_GPU, 'gpu_ids': GPU_IDS, 'global_batch_size': PAPER_20D.batch_size})
models, histories, results = {}, {}, {}
for name, factory in factories.items():
    model = factory()
    if MULTI_GPU:
        model.to(DEVICE)
        model = torch.nn.DataParallel(model, device_ids=GPU_IDS)
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

下载 `artifacts/kaggle_20d/`，把 `all_model_metrics.csv`、`test_predictions.parquet`、训练曲线和 checkpoint 放入本地项目，再用 `scripts/analyze_predictions.py` 从保存的预测重建十分位组合、换手、交易成本与分时段结果。不要把原始数据或 checkpoint 提交到 Git。"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    path = Path('notebooks/price_trends_replication.ipynb')
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


if __name__ == '__main__':
    main()
