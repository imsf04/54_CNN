"""Download the course-cited archive and verify that data files are present."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.adapters.data_source import download_and_extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--url", default=None)
    args = parser.parse_args()
    extracted = download_and_extract(args.output, args.url)
    dat = sorted(extracted.rglob("*.dat"))
    feather = sorted(extracted.rglob("*.feather"))
    print(f"extracted={extracted}")
    print(".dat files:", *dat, sep="\n  ")
    print(".feather files:", *feather, sep="\n  ")
    if not dat or not feather:
        raise SystemExit("解压后未找到 .dat 和 .feather 文件")


if __name__ == "__main__":
    main()
