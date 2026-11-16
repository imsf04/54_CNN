"""Create a Kaggle-safe archive with sanitized course filenames."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path


def safe_name(name: str) -> str:
    return re.sub(r"[\[\]]", "_", name)


def main() -> None:
    source = Path("data/raw/extracted/monthly_20d")
    output = Path("data/price-trends-20d-kaggle.zip")
    files = sorted(path for path in source.iterdir() if path.is_file() and not path.name.startswith("._"))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, f"monthly_20d/{safe_name(path.name)}")
            print(path.name)
    print(f"created {output} from {len(files)} files")


if __name__ == "__main__":
    main()
