# 训练性能剖析

## 目的

12,000 样本、4 epoch 的预算是完整实验前的吞吐与协议验收，不是统计结论。本文记录在 V100 上对生产训练管线做的单变量测量，用于回答三个问题：数据加载是否是瓶颈、batch 大小对吞吐的影响、评估推理能否用更大的 batch。

## 环境

- GPU：Tesla V100-PCIE-32GB，驱动 580.65.06
- PyTorch 2.8.0+cu128，CUDA 可用
- 数据：482,925 训练样本、206,969 验证样本、1,491,887 测试样本；训练集 mean `0.10689710380321306`、std `0.3089823829956947`
- 管线：`IterableImageDataset` + `DataLoader(batch_size=None)`，进程内 memmap 读取，AMP fp16

## 方法

`scripts/profile_training.py` 用真实管线构造 loader 与 `cnn_3blocks` 模型，先热身，再对每个配置计时若干步，同时用 `nvidia-smi` 采样 GPU 利用率，记录 samples/s、ms/step 与 loader 等待时间。配置逐项改变：batch 大小、channels-last、`torch.compile`、训练精度；评估单独测 fp32/AMP 与 batch 128/1024。

评估批大小的等价性用 `scripts/verify_eval_batch.py` 在 `seed_7_cnn_20d.pt` 上验证：同一个验证集分别用 batch 128 与 1024 推理，比较概率的最大与平均绝对差。

## 结果

训练：

| 配置 | batch | 精度 | samples/s | ms/step | GPU 利用率 | loader 等待 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 128 | AMP | 10,332 | 12.4 | 中位 69%，p10 37% | 中位 2.0 ms |
| baseline 复测 | 128 | AMP | 9,560 | 13.4 | 中位 59% | — |
| channels-last | 128 | AMP | 10,909 | — | — | — |
| torch.compile | 128 | AMP | 10,621 | — | 中位 44% | — |
| batch 256 | 256 | AMP | 12,740 | — | 中位 80%，均值 85.7% | — |
| batch 512 | 512 | AMP | 13,099 | — | 中位 79% | — |
| batch 1024 | 1024 | AMP | 13,572 | — | 中位 81% | — |

评估（206,969 行验证集）：

| batch | 精度 | samples/s | 秒 |
| --- | --- | --- | --- |
| 128 | fp32 | 11,479 | 18.0 |
| 128 | AMP | 12,516 | — |
| 1024 | fp32 | 20,312 | 10.2 |

batch 128 与 1024 的概率差：最大 `1.10e-06`，平均 `1.43e-07`，属于浮点舍入。

全量 epoch 成本：482,925 行训练约 47 秒；加上 1,491,887 行测试预测，单次 epoch 总墙钟 193 秒。

## 结论

数据加载不是瓶颈：loader 等待中位数 2 ms，约占每步的 16%。batch 128 时 GPU 利用率中位数 59–69%，是论文协议的固定成本（小模型、每步 Python 与 kernel 启动开销），不是数据供给不足。batch 增大到 256 后利用率升到 80% 左右，1024 时吞吐 13,572 samples/s，继续增大收益递减。channels-last 与 `torch.compile` 的差别在测量噪声内，不引入。

据此固定两点：训练保持 batch 128（与论文一致），评估使用 batch 1024（浮点等价，验证集推理从 18.0 秒降到 10.2 秒）。V100 的计算能力正常：matmul 实测 12.6–13.4 TFLOP/s。

## 复现

```bash
python scripts/profile_training.py --data-dir /root/autodl-tmp/data/extracted --device cuda
python scripts/verify_eval_batch.py --data-dir /root/autodl-tmp/data/extracted --checkpoint /root/autodl-tmp/full_seeds/seed_7_cnn_20d.pt
```
