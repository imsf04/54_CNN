from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    cells = [
        md(
            """# （Re-）Imag（in）ing Price Trends：20-day CNN 复现

本 Notebook 只负责组织实验；数据加载、时间切分、训练、指标和 Grad-CAM 已封装在 `src/price_trends` 包中。所有模型参数来自 Jiang、Kelly 和 Xiu（2023），课程时间窗口来自 `Final Project.pdf` 第 5.2 节。

参考文献：
- [Jiang, Kelly, and Xiu (2023)](https://doi.org/10.1111/jofi.13268)
- [课程指定数据](https://dachxiu.chicagobooth.edu/download/img_data.zip)
- [项目参数来源](../docs/references/README.md)"""
        ),
        code(
            """from pathlib import Path\nimport os\nimport sys\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport torch\nfrom torch.utils.data import DataLoader\n\nROOT = Path.cwd()\nif not (ROOT / 'src').exists():\n    ROOT = Path('..').resolve()\nsys.path.insert(0, str(ROOT / 'src'))\n\nfrom price_trends.adapters.runtime import select_device\nfrom price_trends.data import ImageDataset, load_real_data, temporal_split\nfrom price_trends.models import PriceTrendCNN\nfrom price_trends.references import PAPER_20D, REPRODUCIBILITY_SEED\nfrom price_trends.services.evaluation import evaluate_model\nfrom price_trends.services.interpretation import grad_cam\nfrom price_trends.services.training import fit, set_seed\n\nset_seed(REPRODUCIBILITY_SEED)\nDEVICE = select_device(os.getenv('PRICE_TRENDS_DEVICE', 'auto'))\nDATA_DIR = Path(os.getenv('PRICE_TRENDS_DATA_DIR', ROOT / 'data/raw/extracted'))\nOUTPUT_DIR = ROOT / 'artifacts/notebook'\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\nMAX_EPOCHS = 100  # 仅作为运行上限；实际停止规则是论文规定的 patience=2\nprint({'data_dir': str(DATA_DIR), 'device': str(DEVICE), 'paper_config': PAPER_20D})"""
        ),
        md(
            """## 1. 数据检查与课程时间切分\n\n课程要求使用 64×60 的 20-day 图像；标签为 1（未来收益为正）、0（非正），2 表示缺失。删除缺失标签时，图像和标签使用相同的原始行索引，避免错位。development 使用 1993–1999，2000 年以后只用于 out-of-sample。"""
        ),
        code(
            """has_real_data = DATA_DIR.exists() and any(DATA_DIR.rglob('*.dat')) and any(DATA_DIR.rglob('*.feather'))\nif has_real_data:\n    bundle = load_real_data(DATA_DIR, target='Retx_20d_label', horizon='20d', return_column='Retx_20d')\n    splits = temporal_split(bundle)\n    display(pd.DataFrame({name: {'样本数': len(part.labels), '正类比例': part.labels.mean(), '最早日期': str(part.dates.min()), '最晚日期': str(part.dates.max())} for name, part in splits.items()}).T)\nelse:\n    bundle = None\n    splits = None\n    print('未找到真实数据。请先运行：uv run python scripts/download_data.py --output data/raw')"""
        ),
        md(
            """## 2. 训练论文基线 CNN\n\n20-day CNN 的结构为 3 个 building blocks，filters 为 64/128/256；优化器、学习率、batch size、dropout、初始化和 early stopping 均由论文配置提供。`MAX_EPOCHS` 只是计算上限，不改变论文的停止规则。"""
        ),
        code(
            """if splits is not None:\n    loaders = {\n        name: DataLoader(ImageDataset(part), batch_size=PAPER_20D.batch_size, shuffle=name == 'train')\n        for name, part in splits.items()\n    }\n    model = PriceTrendCNN()\n    history = fit(model, loaders['train'], loaders['val'], epochs=MAX_EPOCHS, device=DEVICE, checkpoint=OUTPUT_DIR / 'cnn_20d.pt')\n    print('最佳模型已恢复；训练轮数:', len(history))\nelse:\n    loaders = model = history = None"""
        ),
        md("""## 3. Out-of-sample 评估"""),
        code(
            """if model is not None:\n    metrics = evaluate_model(model, splits['test'], loaders['test'], DEVICE)\n    display(pd.DataFrame([metrics], index=['CNN 20-day']))\n    pd.DataFrame([metrics]).to_json(OUTPUT_DIR / 'test_metrics.json', orient='records', indent=2)\nelse:\n    metrics = None"""
        ),
        code(
            """if history is not None:\n    rows = []\n    for row in history:\n        rows.append({'epoch': row['epoch'], 'train_loss': row['train']['loss'], 'val_loss': row['val']['loss'], 'train_accuracy': row['train']['accuracy'], 'val_accuracy': row['val']['accuracy']})\n    history_frame = pd.DataFrame(rows)\n    display(history_frame)\n    history_frame.plot(x='epoch', y=['train_loss', 'val_loss'], marker='o', title='训练与验证交叉熵')\n    plt.show()"""
        ),
        md("""## 4. Grad-CAM：查看 CNN 使用的图像区域"""),
        code(
            """if model is not None:\n    model.eval()\n    sample_x, sample_y = next(iter(loaders['test']))\n    sample_x = sample_x[:4].to(DEVICE)\n    with torch.no_grad():\n        probabilities = torch.softmax(model(sample_x), dim=1)[:, 1].cpu()\n    fig, axes = plt.subplots(len(sample_x), 2, figsize=(8, 3 * len(sample_x)))\n    if len(sample_x) == 1:\n        axes = [axes]\n    for row, (image, probability) in enumerate(zip(sample_x, probabilities)):\n        heatmap = grad_cam(model, image.unsqueeze(0), class_index=int(probability >= 0.5)).detach().cpu()\n        axes[row][0].imshow(image[0].detach().cpu(), cmap='gray')\n        axes[row][0].set_title(f'原图，P(up)={probability:.3f}')\n        axes[row][1].imshow(image[0].detach().cpu(), cmap='gray')\n        axes[row][1].imshow(torch.nn.functional.interpolate(heatmap[None, None], size=(64, 60), mode='bilinear', align_corners=False)[0, 0], cmap='magma', alpha=0.55)\n        axes[row][1].set_title('Grad-CAM')\n    plt.tight_layout()\n    plt.show()"""
        ),
        md("""## 5. 架构消融\n\n论文的敏感性分析建议改变 filters 或 blocks。这里保持数据切分、损失、学习率、batch size 和 early stopping 不变，只改变架构，并把结果写入 JSON。"""),
        code(
            """if splits is not None:\n    ablations = {\n        'baseline_64_128_256': (64, 128, 256),\n        'fewer_blocks_64_128': (64, 128),\n        'four_blocks_64_128_256_512': (64, 128, 256, 512),\n    }\n    ablation_rows = []\n    for name, filters in ablations.items():\n        candidate = PriceTrendCNN(filters=filters)\n        candidate_history = fit(candidate, loaders['train'], loaders['val'], epochs=MAX_EPOCHS, device=DEVICE, checkpoint=OUTPUT_DIR / f'{name}.pt')\n        candidate_metrics = evaluate_model(candidate, splits['test'], loaders['test'], DEVICE)\n        ablation_rows.append({'model': name, 'epochs': len(candidate_history), **candidate_metrics})\n    ablation_frame = pd.DataFrame(ablation_rows)\n    display(ablation_frame)\n    ablation_frame.to_csv(OUTPUT_DIR / 'ablation_metrics.csv', index=False)\nelse:\n    ablation_frame = None"""
        ),
        md(
            """## 6. 报告输出\n\n请将 `artifacts/notebook/` 中的指标表、训练曲线和 Grad-CAM 图保存到 `reports/replication_report.md` 对应章节。结论只围绕 out-of-sample 指标、portfolio 指标、可视化和消融结果展开。"""
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Price Trends Final Project",
                "language": "python",
                "name": "price-trends-final-project",
            }
        },
    )
    path = Path("notebooks/price_trends_replication.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


if __name__ == "__main__":
    main()
