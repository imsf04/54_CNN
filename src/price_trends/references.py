"""Paper- and course-sourced constants used by the replication.

The values in this module are intentionally kept in one place so the
notebook does not contain unexplained architecture or training choices.
"""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = (
    "https://economics.yale.edu/sites/default/files/2023-11/"
    "The%20Journal%20of%20Finance%20-%202023%20-%20JIANG%20-%20Re%25E2%2580%2590%20"
    "Imag%20in%20ing%20Price%20Trends_0.pdf"
)
PAPER_DOI_URL = "https://doi.org/10.1111/jofi.13268"
COURSE_PDF = "Final Project.pdf, section 5.2"
OFFICIAL_DATA_URL = "https://dachxiu.chicagobooth.edu/download/img_data.zip"
MIRROR_DATA_URL = "https://www.dropbox.com/s/njehqednn8mycze/img_data.zip?dl=1"


@dataclass(frozen=True)
class CNNReferenceConfig:
    """Parameters reported for the paper's 20-day image CNN."""

    image_height: int = 64
    image_width: int = 60
    blocks: int = 3
    first_filters: int = 64
    filter_growth: int = 2
    convolution_kernel: tuple[int, int] = (5, 3)
    pooling_kernel: tuple[int, int] = (2, 1)
    first_vertical_stride: int = 3
    first_vertical_dilation: int = 2
    leaky_relu_slope: float = 0.01
    dropout: float = 0.50
    learning_rate: float = 1e-5
    batch_size: int = 128
    early_stopping_patience: int = 2


PAPER_20D = CNNReferenceConfig()

# The course handout changes the paper's eight-year development window to
# 1993--1999 and holds 2000 onward out of sample. This is a course constraint,
# not an invented model choice.
COURSE_DEV_END_YEAR = 1999
COURSE_TEST_START_YEAR = 2000
REPRODUCIBILITY_SEED = 7  # bookkeeping only; not a model hyperparameter
