# 术语表

- **20-day 图像**：用前 20 个交易日价格、成交量、移动平均线编码的 64×60 灰度图。
- **未来收益方向标签**：未来收益为正记为 1，非正记为 0，原始缺失值记为 2。
- **development 样本**：课程要求的 1993–1999 年样本，用于随机 70/30 train/validation 切分。
- **out-of-sample 样本**：2000 年及以后样本，只用于最终测试。
- **PriceTrendCNN**：论文 20-day 规格的三层 CNN，filters 为 64/128/256。
- **horizon**：未来收益窗口；本课程数据使用 20-day 图像，也提供 5d、20d、60d 未来收益标签。
- **long-short portfolio**：按预测概率分组，做最高概率组多头、最低概率组空头的日期级组合。
- **decile long-short portfolio**：按每个日期的预测概率排序，取最高 10% 与最低 10% 形成的等权多空组合。
- **训练集标准化统计量**：仅由 development 中的 train split 计算的像素 mean/std，并固定应用于 validation 和 out-of-sample。
- **Grad-CAM**：显示最后卷积层对某个类别预测贡献区域的热图。
