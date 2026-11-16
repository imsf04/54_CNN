# Kaggle T4 ×2 资源利用与峰值控制研究

> 研究范围：Kaggle Notebook 的 T4 双 GPU 会话，以及本仓库当前的价格趋势图像训练链路。资料只采用 Kaggle、PyTorch、NumPy、NVIDIA 一手文档；本文不修改生产代码。仓库状态观察时间：2026-11-17（Asia/Singapore）。

## Goal

在 Kaggle 的 T4 ×2 会话中，尽可能同时利用 CPU、RAM 和两张 GPU，并避免因数据预取、进程复制或一次性物化数据而超过内存/显存峰值；同时解释“数据加载单元格 CPU/GPU 利用率低”的原因。

## Main contradiction

吞吐需要并行预取、页锁定内存、较大的 batch 和两个 GPU；峰值控制则要求限制 worker 数、预取深度、batch 和常驻副本。当前仓库还把 `.dat` 的有效样本从临时 `memmap` 转成 NumPy 数组并最终 `concatenate`，因此“使用了 memmap”并不等于端到端的 out-of-core 读取。

## Evidence

### Kaggle 与硬件边界

- Kaggle 官方 Notebook 文档说明 Notebook 在容器会话中运行，GPU 通过 **Settings → Accelerator → GPU** 选择；文档没有把 “T4 ×2” 的 CPU/RAM 配额承诺为稳定 API。每次运行应打印 `nvidia-smi`、`torch.cuda.device_count()`、设备名称和可用主机内存来核验实际会话：[Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)。
- Kaggle 官方仓库的讨论显示 UI 中的 “GPU T4 ×2” 与 machine shape/API 映射并非稳定契约，因此双卡数量必须在运行时确认：[kaggle-cli issue #1196](https://github.com/Kaggle/kaggle-cli/issues/1196)。
- NVIDIA 规格页给出单张 T4 为 16 GB GDDR6；两张卡的显存是两个独立设备，不能因为总量约 32 GB 就当作一个可直接分配的 32 GB 地址空间：[NVIDIA Tesla T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)。

### DataLoader、CPU/RAM 与 host→device 传输

- `DataLoader(num_workers=0)` 在主进程加载；`num_workers>0` 才会使用 worker 子进程。`pin_memory=True` 会把返回 batch 放入 page-locked host memory，PyTorch 文档说明这能加快向 CUDA 设备的复制；`prefetch_factor` 控制每个 worker 预取的 batch 数；`persistent_workers=True` 使 worker 跨 epoch 保持，不必每轮重启：[PyTorch `torch.utils.data`](https://docs.pytorch.org/docs/2.15/data.html)。
- PyTorch 同一文档警告，多进程 DataLoader 可能让 worker 复制父进程中被访问的 Python 对象，整体 CPU 内存消耗可能随 worker 数近似增长。因此 `num_workers`、`prefetch_factor`、batch size 共同决定 RAM 峰值，不能只看模型显存。
- NumPy `memmap` 允许只访问大文件的一部分而不把整个文件读入内存，适合将样本索引化后按需切片；它本身不提高 GPU 利用率，随机访问仍可能受磁盘 I/O 限制：[NumPy `memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)。
- 本仓库的现状是：`load_real_data()` 用 `np.memmap` 打开每年 `.dat`，随后 `np.asarray(images[valid_indices])`、`np.concatenate(image_parts)`，`ImageDataset` 又建立 float32 torch tensor；`run_experiments.py` 的 DataLoader 保持默认 `num_workers=0`、未设置 `pin_memory`。这些本地观察分别见 [`src/price_trends/data.py`](../../src/price_trends/data.py) 和 [`scripts/run_experiments.py`](../../scripts/run_experiments.py)。

### 两张 GPU、AMP 与利用率观测

- PyTorch CUDA 语义是异步的：CPU 发起 kernel 后可以继续执行，只有需要结果或显式同步时才等待。由此，数据准备/拷贝、很短的 kernel、以及训练循环中的主机端读取都可能造成 GPU 空转或利用率呈脉冲状；应以时间线和端到端 step time 判断，而不是单次采样值：[PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)。
- PyTorch CUDA 文档建议多 GPU batched workloads 默认使用 `DistributedDataParallel`；单机应采用每进程/每 GPU 的方式，CUDA GPU 通常使用 NCCL backend，而不是单进程 `DataParallel`：[PyTorch CUDA semantics — multi-GPU](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
- `DistributedDataParallel` 不会自动把两张卡的显存合并；每个 rank 有自己的模型和 optimizer 状态，需用 `DistributedSampler` 让 rank 读取不同样本分片。若每 rank 启动 worker，总 worker 数还要乘以 rank 数：[DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)。
- AMP 的 autocast 会把适合的算子转为低精度，GradScaler 用于 float16 梯度缩放；T4 具有 Tensor Cores，适合先实测 FP16 AMP 的吞吐/显存收益：[PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/2.15/accelerator/amp.html)、[NVIDIA Tesla T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)。
- `nvidia-smi` 的 utilization 是采样窗口内 GPU 忙碌比例（`utilization.gpu` 反映 kernel 执行时间，memory utilization 反映显存读写忙碌程度），不是“已占用显存比例”。所以数据加载阶段看到低 GPU util 并不矛盾：[NVIDIA `nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

## Decision

### 推荐的 Kaggle 运行顺序

1. **先做资源核验。** 在训练前记录 `nvidia-smi`、`torch.cuda.device_count()`、每张卡名称/显存、CPU 核数、主机总/可用 RAM；若不是两张 T4，停止把“双卡结果”当作已成立。
2. **先控制数据峰值，再加并行。** 将 Dataset 改成“索引清单 + 每个 worker 按需打开 memmap 并切片”，避免 `np.concatenate` 全量物化；每个 rank/worker 不要缓存整份父进程数组。当前数据文件单个可达数百 MB，仓库 `data/raw/extracted` 的 `.dat` 合计约 7.86 GB（本地观测），全量转 float32 会进一步放大峰值。
3. **单卡基线参数。** 从 `batch_size` 保守值、`num_workers=2`、`prefetch_factor=1`、`pin_memory=True`、`persistent_workers=True` 开始；用主机 RSS、GPU 显存峰值和 step time 逐项增加 worker 或 batch。内存紧张时先把 `persistent_workers=False`、`prefetch_factor=1`，再减少 worker/batch；不要盲目按 CPU 核数开 worker。
4. **让拷贝与计算重叠。** 对 pinned batch 使用 `x.to(device, non_blocking=True)`（前提是 batch 确实由 pinned memory 提供）；只在 GPU 训练时启用 pinning，因为 pinned memory 会消耗主机 RAM。把预处理放入 worker，避免在训练循环中做昂贵 Python/NumPy 操作（[PyTorch DataLoader memory pinning](https://docs.pytorch.org/docs/2.15/data.html)）。
5. **双卡采用 DDP。** 使用 `torchrun --nproc_per_node=2`（或 Kaggle 会话等价的两进程启动），每个进程绑定一张 GPU、初始化 NCCL、使用 `DistributedSampler`；验证每个 rank 的 `LOCAL_RANK`、设备、样本数和 checkpoint 只保存一次。DataLoader worker 数按“每 rank”配置并监控总 RAM（[PyTorch DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)）。
6. **再启用 AMP。** 训练前向/损失放进 `torch.autocast("cuda", dtype=torch.float16)`，使用 `torch.amp.GradScaler("cuda")`；先确认收敛与指标，再逐步增大 micro-batch。AMP 主要减少显存和提高算力吞吐，不能修复输入流水线或极小模型造成的低利用率（[PyTorch AMP](https://docs.pytorch.org/docs/2.15/accelerator/amp.html)）。
7. **用时间窗口解释低利用率。** 数据加载单元格通常只在 CPU 读文件、解码、整理 batch，GPU 尚未执行 kernel，因此 GPU util 低是预期现象；训练中若 GPU 呈周期性尖峰，优先检查 worker/预取、磁盘吞吐、host→device 拷贝和同步点。用 `nvidia-smi dmon`/采样日志配合 step time、主机 RSS、显存峰值；不要以单个瞬时 util 数字判定是否“浪费 GPU”。

### 可接受的停止条件

- 任一 worker 配置使主机可用 RAM 持续下降或出现 OOM：回退 `num_workers`/`prefetch_factor`/batch，优先保留 out-of-core Dataset。
- 两卡 DDP 的端到端 step time 没有改善、通信/输入等待占比升高：保留单卡基线，记录证据，不把“启用双卡”写成性能提升。
- AMP 改变验证指标或出现 NaN：回退 FP32 或缩小 AMP 范围；不要为了利用率牺牲复现实验契约。

## Next step

本轮已先落地低风险、可回退的输入流水线修复：`load_real_data(max_samples=...)` 在采样工程运行中按年度文件使用最多 4 个受控线程，并先对采样索引排序后读取 memmap，避免单线程随机页访问；正式 Notebook 还会在每次运行前同步仓库到 `origin/main`，把 Kaggle 只读输入 staging 到本地工作盘，并配置有限的 DataLoader worker、prefetch 和 pinned memory。一次本地 120,000 样本加载基准由约 26 秒降至约 4 秒；这只是本机 I/O 基准，不代表 Kaggle 端到端加速。当前 Notebook 额外提供 `DataParallel` 作为双卡即时利用率检查；正式论文级多卡结果仍应使用两进程 DDP，不能把 DataParallel 的吞吐直接写成最终结论。

## Reopen condition

当 Kaggle 会话实际暴露的 GPU 型号/数量、CPU/RAM 配额，或 PyTorch/CUDA 版本改变；或者 benchmark 证明输入流水线已饱和而 GPU 仍低利用率时，重新核对官方文档与时间线，更新参数建议。

