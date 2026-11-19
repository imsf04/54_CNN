# 扎实复现线工作记录

## Goal

在进入远端 GPU 全量运行前，完成课程 5.2 要求的本地可审计复现：训练集标准化、论文式 decile 组合评估、固定预算架构消融和多样本 Grad-CAM。

## Main contradiction

资源有限要求快速迭代，但一次短训练或一次 seed 不能支撑架构和投资组合结论。解决方式是固定协议、保留每次原始产物，并把远端 GPU 延后到本地链路通过后。

## Evidence

- 协议记录：`docs/adr/0001-rigorous-replication-protocol.md`。
- 本地数据：20-day 图像，12,000 年度均衡样本；train/validation/test 为 2,175/933/8,880。
- 训练集标准化：mean `0.10722509578544061`，std `0.3093992156086432`；统计量只由 train split 计算。
- 固定预算：seeds 7、17、27；每个配置 4 个完整 epoch；每个配置均实际运行 4 epoch。
- 配置：Logistic baseline、2/3/4 blocks、narrow filters、wide filters。
- 评估：top/bottom 10% decile 为主指标，top/bottom 20% 为敏感性指标；均未计交易成本。
- Grad-CAM：最佳本地 CNN 为 `cnn_wide`, seed 17；TP/TN/FP/FN 各 4 张，共 16 张。
- 产物目录：`artifacts/local_ablations_normalized_v2/`，包括 `results.json`、`results.csv`、18 个 checkpoint、`gradcam_examples.json` 和 `gradcam_16_samples.png`。
- 验证：6 个 pytest 通过，ruff 通过，Grad-CAM 网格完成视觉检查。

## Decision

本轮结果只作为本地协议验收和架构筛查，不宣称 CNN 优势或稳定投资价值。3-seed ROC-AUC 均值范围约为 0.4924–0.5083，组合 Sharpe 对 seed 敏感，说明需要谨慎解释。Grad-CAM 固定使用 `cnn_3blocks`, seed 7，不使用测试集选择模型。

## Next step

1. 复核 decile 组合实现与论文口径，补充成本敏感性设计。
2. 将标准化统计量和 decile 指标写入 GPU Notebook 输出协议。
3. 远端 GPU 空闲后，运行固定协议的全量 20-day 实验。
4. 比较全量结果与本地 3-seed 结果，保留差异和失败记录。

## Reopen condition

若数据文件、课程切分、GPU/CUDA/PyTorch 版本、标签定义或组合分组口径发生变化，必须重新打开本记录并更新 ADR；不得直接沿用旧结果。

## Remote preflight observation — 2026-11-18

Read-only SSH check of `10.10.20.20` found both GPUs occupied: GPU 0 at 95% utilization with 25,408 MiB used, GPU 1 at 1% with 9,832 MiB used. Other users had active sessions. The host shell did not expose `python` on PATH. Formal training was not started.

## 全量 20-day 运行 — 2026-11-18

- 单次全量运行：Kaggle kernel `imsfp04/price-trends-rigorous-20-day-replication`，revision `01eb696`，2,181,781 个样本，train/validation/test 为 482,925/206,969/1,491,887；训练集 mean `0.10689710380321306`，std `0.3089823829956947`；产物在 `artifacts/kaggle_output_1118/`。
- 三 seed 全量运行：V100 上执行 `scripts/run_full_experiments.py --seeds 7 17 27 --epochs 100`，混合精度、lazy memmap 读取；产物在 `artifacts/full_seeds/`，三个 seed 的测试预测用于逐 seed 与概率平均的汇总。
- 年化口径：课程横截面为月度（240 个日期观测，每期持有 20 个交易日的收益），`long_short_metrics` 默认 `periods_per_year=12`；按 252 计算的旧数字不再引用。
- 三 seed 结果（mean±sd）：Logistic AUC `0.5251±0.0008`、Sharpe `1.94±0.18`；MLP AUC `0.5275±0.0009`、Sharpe `1.60±0.12`；CNN AUC `0.5300±0.0038`、Sharpe `1.70±0.35`。三个 seed 概率平均后的 Sharpe 为 1.99/1.65/1.88，扣除 10 bps 成本为 1.46/1.29/1.51。
- 判断：CNN 的 AUC 优势只有 0.0025–0.0049，小于自身 seed 标准差 0.0038；组合 Sharpe 的 seed 标准差 0.35 约为其他模型两倍，单次全量运行的 CNN 高收益不能作为结论。
- 分时段：三个模型在 2013–2019 都衰减，AUC 0.511–0.516，月均组合收益 0.45–0.65%。
- 全量架构消融完成：15 个 run（5 个配置 × 3 个 seed），每个 8 epoch、训练 batch 128、评估 batch 1024，共约 3.5 小时；结果在 `artifacts/full_ablations/results.json`，日志与命令在 `artifacts/full_ablations/`。均值±标准差（AUC / Sharpe）：2 blocks `0.528±0.001 / 1.75±0.05`，3 blocks `0.529±0.004 / 1.72±0.21`，4 blocks `0.528±0.003 / 1.48±0.26`，narrow `0.529±0.001 / 1.44±0.08`，wide `0.529±0.003 / 1.67±0.14`。五个变体的 AUC 差距在 seed 波动内，深度与宽度不产生可靠排序；报告第 5 节已改为全量消融表。
- 决定：报告以三 seed 均值与 seed 平均后的组合为主，单次 Kaggle 运行只作为独立重复；Grad-CAM 与遮挡实验使用该单次运行的 checkpoint。

## 训练吞吐剖析 — 2026-11-19

- 环境与数据：V100-32GB、驱动 580.65.06、torch 2.8.0+cu128；482,925 训练 / 206,969 验证 / 1,491,887 测试样本。
- 生产管线（`IterableImageDataset` + `DataLoader(batch_size=None)`，AMP）在 batch 128 下为 9,560–10,332 samples/s、12.4–13.4 ms/step，GPU 利用率中位 59–69%，loader 等待中位 2.0 ms，约占每步 16%。
- batch 256/512/1024 吞吐为 12,740/13,099/13,572 samples/s，利用率升到 79–81%；channels-last 10,909、`torch.compile` 10,621，与 baseline 的差异在测量噪声内。
- 评估 batch 128 fp32 为 18.0 秒/20.7 万行，batch 1024 为 10.2 秒；两者概率最大差 `1.10e-06`、平均 `1.43e-07`。
- 决定：训练保持 batch 128（论文协议），评估使用 batch 1024；12,000 样本/4 epoch 的定位是吞吐与协议验收预算，不作为统计结论。详见 `docs/research/training_profile.md` 与 `artifacts/training_profile.json`。
