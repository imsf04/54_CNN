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
