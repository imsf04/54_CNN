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

## 4. 结果

| Model | Accuracy | ROC-AUC | AP | Cumulative long-short return | Annualized volatility | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Logistic | 0.4965 | 0.5058 | 0.5067 | 2.4704 | 1.2206 | 1.6425 |
| MLP | 0.5087 | 0.5236 | 0.5305 | 18.8704 | 1.1127 | 3.4030 |
| CNN | 0.4941 | 0.5045 | 0.5072 | -0.8559 | 1.4666 | -0.5975 |

这些结果来自短训练和小样本工程验证，不能支持论文级性能结论。多空组合未计交易成本，且只有 240 个日期级组合观测。

## 5. 解释与限制

Grad-CAM 和架构消融入口已实现，但本轮没有把真实样本图和完整消融表作为结论写入报告。下一轮应在固定训练预算下运行 baseline、CNN 深度、滤波器数量和 Grad-CAM 样本分析。

当前结果只能回答“真实数据链路和三模型服务是否可以运行”。不能回答 CNN 是否优于 baseline，也不能证明预测信号具有稳定投资价值。最终实验需要全量 20-day 数据、预先固定的训练预算和独立样本外报告。

## 6. 可复现性

- 依赖由 `uv.lock` 固定；
- 清华源为默认 uv index，阿里源为显式 fallback；
- 随机种子由训练服务统一设置；
- 原始数据、checkpoint 和实验产物不提交到 Git；
- 运行结果保存在 `artifacts/data_audit_20d.json` 和 `artifacts/experiments_20d/results.json`。

## 7. AI 使用说明

AI 用于代码组织、错误定位、文档整理和可复现实验辅助。数据文件清单、字段审计、模型运行和结果解释由项目成员在本地环境核验。
