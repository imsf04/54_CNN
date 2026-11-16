# （Re-）Imag（in）ing Price Trends

这是 MSDM5054 Final Project 的中文复现项目。实现以 Jiang、Kelly、Xiu（2023）论文和课程 PDF 第 5.2 节为唯一方法依据，使用论文报告的 20-day CNN 结构，不把烟雾测试当作实验结果。

## 目录

```text
src/price_trends/
├── domain/       对齐的数据实体
├── adapters/     本地、Kaggle、Colab 路径与设备适配
├── models/       论文结构的 CNN
└── services/     训练、评估、Grad-CAM
notebooks/        中文复现 Notebook
scripts/          下载、实验、Notebook 生成
docs/references/  论文 PDF、文本和参数来源
reports/          中文报告框架
data/raw/         下载的原始压缩包与解压文件（不提交 Git）
```

## 真实数据

课程指定数据约 567 MB（压缩包），官方地址是 [Chicago Booth archive](https://dachxiu.chicagobooth.edu/download/img_data.zip)，备用镜像见 [Dropbox](https://www.dropbox.com/s/njehqednn8mycze/img_data.zip?dl=1)。下载并解压：

```bash
uv sync
uv run python scripts/download_data.py --output data/raw
```

如果官方地址较慢，可以显式指定镜像：

```bash
uv run python scripts/download_data.py --output data/raw --url "https://www.dropbox.com/s/njehqednn8mycze/img_data.zip?dl=1"
```

也可以设置 `PRICE_TRENDS_DATA_DIR`，或者在 Kaggle/Colab 中使用 `/kaggle/input/price-trends`、`/content/price-trends`。

## 运行真实实验

```bash
uv run python scripts/run_experiments.py \
  --data-dir data/raw/extracted \
  --horizons 20d \
  --output artifacts/experiments
```

首次检查完整链路可以加 `--max-samples 2000 --max-epochs 1`；这只限制检查规模，不改变正式实验的论文参数。

课程数据是 20-day 图像；如果 Feather 标签同时提供 `Retx_5d_label` 和 `Retx_60d_label`，可以在同一图像上运行：

```bash
uv run python scripts/run_experiments.py --data-dir data/raw/extracted --horizons 5d 20d 60d
```

这里的 5d/20d/60d 指未来收益标签 horizon；课程压缩包只提供 20-day 图像，因此三组实验共享 20-day 图像输入，不能误写成三种不同图像。

脚本使用论文的 Adam、`1e-5` 学习率、batch size `128`、50% dropout、Xavier 初始化和 validation-loss patience `2`。课程要求的 1993–1999 / 2000+ 时间窗口在 `temporal_split` 中固定实现。

## Notebook 与报告

```bash
uv run python scripts/build_notebook.py
jupyter notebook notebooks/price_trends_replication.ipynb
```

Notebook 用中文 Markdown 组织数据检查、训练、out-of-sample 指标、训练曲线、Grad-CAM 和架构消融。报告骨架在 [reports/replication_report.md](reports/replication_report.md)，参数出处在 [docs/references/README.md](docs/references/README.md)。

## 检查

```bash
uv run pytest -q
uv run ruff check src scripts tests
```

没有真实数据时可以运行合成 smoke test，但它只验证代码链路：

```bash
uv run python scripts/smoke_train.py --output artifacts/smoke
```

The project configures the Tsinghua PyPI mirror as the default uv index and keeps Aliyun as an explicit fallback. For a one-off Aliyun sync, run:

```bash
uv sync --default-index https://mirrors.aliyun.com/pypi/simple
```
