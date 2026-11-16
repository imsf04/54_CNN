# 参考文献与参数来源

本项目的模型和训练配置只从以下材料读取：

1. Jiang, Kelly, and Xiu, “(Re-)Imag(in)ing Price Trends,” *Journal of Finance* 78(6), 2023。论文 PDF：
   [Yale Economics mirror](https://economics.yale.edu/sites/default/files/2023-11/The%20Journal%20of%20Finance%20-%202023%20-%20JIANG%20-%20Re%25E2%2580%2590%20Imag%20in%20ing%20Price%20Trends_0.pdf)，
   DOI：[10.1111/jofi.13268](https://doi.org/10.1111/jofi.13268)。本仓库保存了下载的 PDF 和文本版本。
2. 课程作业说明：[Final Project.pdf](../../Final%20Project.pdf)，第 5.2 节。课程要求使用 64×60 的 20-day 图像，并将 1993–1999 作为 development、2000 年以后作为 out-of-sample。
3. 课程说明中的数据源：
   [Chicago Booth official archive](https://dachxiu.chicagobooth.edu/download/img_data.zip)；
   [Dropbox mirror](https://www.dropbox.com/s/njehqednn8mycze/img_data.zip?dl=1)。

## 论文参数摘录

| 项目 | 20-day 复现值 | 来源 |
|---|---:|---|
| 输入图像 | 64×60 | 课程 PDF，第 5.2 节 |
| CNN building blocks | 3 | 论文 Appendix，架构说明 |
| convolution kernel | 5×3 | 论文 Appendix，架构说明 |
| max-pooling kernel | 2×1 | 论文 Appendix，架构说明 |
| 第一层 vertical stride | 3 | 论文 Appendix，架构说明 |
| 第一层 vertical dilation | 2 | 论文 Appendix，架构说明 |
| filters | 64, 128, 256 | 论文 Appendix，逐层乘 2 |
| activation | Leaky ReLU，slope 0.01 | 论文 Appendix 注释 20 |
| batch normalization | 每个 block 的 convolution 后、activation 前 | 论文训练部分 |
| fully-connected dropout | 50% | 论文训练部分 |
| weight initializer | Xavier | 论文训练部分 |
| optimizer | Adam | 论文训练部分 |
| initial learning rate | 1×10⁻⁵ | 论文训练部分 |
| batch size | 128 | 论文训练部分 |
| early stopping | validation loss 连续 2 个 epoch 未改善 | 论文训练部分 |

课程 PDF 将时间窗口调整为 1993–1999 / 2000+，因此代码使用课程窗口；论文正文原始设计是 1993–2000 / 2001–2019。这个差异在报告中明确标注。

随机种子 `7` 只用于复现随机切分和 smoke test，不是论文模型超参数。
