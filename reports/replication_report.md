# （Re-）Imag（in）ing Price Trends：20-day CNN 复现报告

## 1. 执行状态

本报告记录第一次真实数据管线实验。课程数据中实际提供的是 20-day 图像，同时包含 5d、20d、60d 的未来收益标签；因此本轮只把 20-day 图像作为输入，分别评估三个 horizon 标签。5d/60d 独立图像文件在本地数据目录中不存在。

## 2. 数据审计

审计命令：

```bash
uv run python scripts/audit_real_data.py data/raw/extracted --output artifacts/data_audit_20d.json
```

结果：

- 27 个年度图像/标签文件对；
- 2,196,994 个样本；
- 每个年度 `.dat` 图像数量与 Feather 行数一致；
- 所有标签文件包含 `Date`、`StockID`、`Retx_20d_label`；
- `Date + StockID` 缺失数为 0；
- `Date + StockID` 重复数为 0；
- `__MACOSX` 元数据文件被排除。

审计只读取每个年度的标签表，并根据文件大小验证图像数量，不把全部 8.8GB 图像一次性加载进内存。

## 3. 实验设置

本次运行使用年度均衡的 12,000 样本工程验证集，CPU、2 个 epoch。该设置用于验证真实数据 loader、切分、模型比较和评估服务，不代表最终论文训练配置。

```bash
uv run python scripts/run_experiments.py \
  --data-dir data/raw/extracted \
  --output artifacts/experiments_20d \
  --horizons 20d \
  --max-epochs 2 \
  --max-samples 12000 \
  --device cpu
```

数据切分结果：2,175 train、933 validation、8,880 out-of-sample test。

## 4. 本地工程验证结果

| Model | Accuracy | ROC-AUC | AP | Cumulative long-short return | Annualized volatility | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Logistic | 0.4965 | 0.5058 | 0.5067 | 2.4704 | 1.2206 | 1.6425 |
| MLP | 0.5087 | 0.5236 | 0.5305 | 18.8704 | 1.1127 | 3.4030 |
| CNN | 0.4941 | 0.5045 | 0.5072 | -0.8559 | 1.4666 | -0.5975 |

这些结果来自短训练和小样本工程验证，不能支持论文级性能结论。多空组合未计交易成本，且只有 240 个日期级组合观测。

## 5. Kaggle 全量 20-day 正式运行

Notebook 运行版本：`066e69c`。本次运行使用 lazy memmap 数据管线，将图像按 batch 从本地 staging 盘读取，没有把全部图像物化到 RAM。

### 数据与产物核验

- 有效样本总数：2,181,781；其中原始审计文件共 2,196,994 行，过滤掉无效二分类标签或无效日期后得到有效样本；
- train / validation / test：482,925 / 206,969 / 1,491,887；
- 测试预测文件行数：1,491,887；
- 测试日期：2000-01-31 至 2019-12-31；
- 测试标签正类比例：0.50587；
- 预测文件包含 Date、StockID、label、return 以及三个模型的 probability；
- Kaggle 下载包：`artifacts/kaggle_20d_artifacts.zip`，包含 checkpoint、history、metrics 和测试预测。

### 正式运行指标

| Model | Accuracy | Balanced accuracy | ROC-AUC | AP | Cumulative long-short return | Annualized volatility | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 0.5177 | 0.5181 | 0.5262 | 0.5264 | 7.6978 | 0.3019 | 7.7013 |
| MLP | 0.5184 | 0.5193 | 0.5292 | 0.5285 | 11.7415 | 0.3671 | 7.4859 |
| CNN | 0.5124 | 0.5143 | 0.5237 | 0.5227 | 3.5442 | 0.2737 | 5.9595 |

三种模型的分类指标都只略高于随机基线，不能据此宣称模型具有稳定预测能力。多空组合指标未计交易成本，且只使用 240 个日期级组合观测；这些结果只能作为课程项目的样本外描述性结果，不能直接解释为可交易策略收益。

### Early stopping 记录

- Logistic：完整 history 26 epochs，最佳 validation loss 在 epoch 24；
- MLP：完整 history 12 epochs，最佳 validation loss 在 epoch 10；
- CNN：完整 history 4 epochs，最佳 validation loss 在 epoch 2，随后连续两轮没有改善而停止。

这与项目采用的两轮 early stopping 规则一致。checkpoint 中保存的是最佳状态，而不是停止时的最后状态。DataParallel checkpoint 的 state dict 带有 `module.` 前缀；在不使用 DataParallel 的本地分析代码中加载时，需要先去掉该前缀或先用相同的 DataParallel 包装模型。

## 6. 解释与限制

初次全量运行时，Grad-CAM 和架构消融入口虽已实现，但没有把真实样本图和完整消融表写入报告；本地固定预算检查见下一节。

当前结果只能回答“真实数据链路和三模型服务是否可以运行”。不能回答 CNN 是否优于 baseline，也不能证明预测信号具有稳定投资价值。最终实验需要全量 20-day 数据、预先固定的训练预算和独立样本外报告。

## 7. 本地固定预算消融与 Grad-CAM

在关闭 Kaggle 远端后，先用本地真实数据做固定预算检查。实验命令为：

```bash
uv run python scripts/run_local_ablations.py \
  --data-dir data/raw/extracted \
  --output artifacts/local_ablations_normalized_v2 \
  --max-samples 12000 \
  --epochs 4 \
  --seeds 7,17,27 \
  --device cpu
```

所有配置使用同一个年度均衡样本上限、同一个时间切分、batch size 128 和 4 个完整 epoch；patience 设为 5，避免早停造成预算不一致。使用 seeds 7、17、27，并保存每个 seed 的原始结果和 checkpoint。训练集像素统计量为 mean=0.107225、std=0.309399，只由 train split 计算，再固定应用到 validation/test。数据切分为 2,175 train、933 validation、8,880 test。

| Configuration | ROC-AUC mean ± sd | AP mean ± sd | Decile Sharpe mean ± sd | Top/bottom 20% Sharpe mean |
|---|---:|---:|---:|---:|
| Logistic baseline | 0.5041 ± 0.0035 | 0.5094 ± 0.0030 | 0.9840 ± 1.2791 | 0.8028 |
| CNN, 2 blocks (64,128) | 0.4924 ± 0.0077 | 0.5001 ± 0.0044 | -0.9758 ± 0.7069 | -0.0648 |
| CNN, 3 blocks (64,128,256) | 0.4933 ± 0.0090 | 0.4990 ± 0.0066 | 0.7379 ± 1.3619 | 0.7430 |
| CNN, 4 blocks (64,128,256,512) | 0.4942 ± 0.0046 | 0.5005 ± 0.0026 | 0.6663 ± 0.7165 | 0.2387 |
| CNN narrow (32,64,128) | 0.5001 ± 0.0077 | 0.5037 ± 0.0064 | -0.2460 ± 0.1009 | 0.1176 |
| CNN wide (128,256,512) | **0.5083 ± 0.0090** | **0.5125 ± 0.0088** | 0.0504 ± 0.9836 | 0.5366 |

3 个 seed 的短预算结果仍然接近随机分类基线，且不同架构的 Sharpe 波动较大；不能据此宣称 CNN 优于 Logistic，也不能把未计成本的组合收益解释为稳定投资价值。组合主指标现在是每个日期 top/bottom 10% 的等权 decile long-short，top/bottom 20% 仅作敏感性分析。

评估器要求每个日期至少有 10 个有效资产，否则直接失败；这避免在小样本 smoke run 中把不完整的 decile 组合误当成正式结果。

Grad-CAM 按预先声明的固定模型（`cnn_3blocks`, seed 7）生成固定 16 张样本：TP、TN、FP、FN 各 4 张，并保存逐样本日期、股票、标签、概率和预测类别。模型选择不使用测试集指标。网格图为 `artifacts/local_ablations_normalized_v2/gradcam_16_samples.png`，metadata 为 `gradcam_examples.json`。这仍然是定性解释，不是因果证据。

原始 JSON、CSV、checkpoint、Grad-CAM metadata 和图像均保存在 `artifacts/local_ablations_normalized_v2/`，便于后续 GPU 实验复核。

## 8. 可复现性

- 依赖由 `uv.lock` 固定；
- 清华源为默认 uv index，阿里源为显式 fallback；
- 随机种子由训练服务统一设置；
- 原始数据、checkpoint 和实验产物不提交到 Git；
- 本地工程验证结果保存在 `artifacts/data_audit_20d.json` 和 `artifacts/experiments_20d/results.json`；
- Kaggle 正式运行产物下载为 `artifacts/kaggle_20d_artifacts.zip`，解压目录被 `.gitignore` 排除，不提交原始数据和大模型产物。

## 9. AI 使用说明

AI 用于代码组织、错误定位、文档整理和可复现实验辅助。数据文件清单、字段审计、模型运行和结果解释由项目成员在本地环境核验。
