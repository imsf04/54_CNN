# Glossary

- **20-day image**: A 64×60 image encoding the prior 20 trading days of a stock's price and volume information.
- **Binary trend label**: The target where 1 means positive subsequent return and 0 means non-positive subsequent return. Missing values are encoded as 2 in the source labels.
- **Development sample**: Observations dated 1993–1999, used for the 70/30 train/validation split.
- **Out-of-sample sample**: Observations dated 2000 onward, held out for final evaluation.
- **PriceTrendCNN**: The three convolution-block baseline model used to classify binary trend labels from 20-day images.
- **Horizon**: The future-return window used to define a target, evaluated independently for 5-day, 20-day, and 60-day experiments.
- **Long-short portfolio**: A date-level portfolio that buys the highest predicted-probability group and sells the lowest group.
- **Grad-CAM**: A class-conditional heatmap showing which spatial regions most influenced a CNN prediction.
