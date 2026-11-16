"""Audit every real image/label year without loading all pixels into memory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def year(path: Path) -> int:
    match = re.search(r"_(\d{4})_(?:images|labels)", path.name)
    if match is None:
        raise ValueError(f"Cannot infer year from {path.name}")
    return int(match.group(1))


def audit(data_dir: Path, horizon: str = "20d") -> dict[str, object]:
    root = data_dir.resolve()
    image_files = {
        year(path): path
        for path in root.rglob("*.dat")
        if "__MACOSX" not in path.parts and not path.name.startswith("._") and horizon in path.name.lower()
    }
    label_files = {
        year(path): path
        for path in root.rglob("*.feather")
        if "__MACOSX" not in path.parts and not path.name.startswith("._") and horizon in path.name.lower()
    }
    if set(image_files) != set(label_files):
        raise ValueError(f"Image/label years differ: images={sorted(image_files)}, labels={sorted(label_files)}")
    years: list[dict[str, object]] = []
    for current_year in sorted(image_files):
        image_path, label_path = image_files[current_year], label_files[current_year]
        image_count, remainder = divmod(image_path.stat().st_size, 64 * 60)
        if remainder:
            raise ValueError(f"Image bytes are not divisible by 64*60: {image_path}")
        labels = pd.read_feather(label_path)
        required = {"Date", "StockID", "Retx_20d_label"}
        missing = required.difference(labels.columns)
        if missing:
            raise KeyError(f"Missing columns in {label_path}: {sorted(missing)}")
        keys = labels[["Date", "StockID"]]
        duplicate_keys = int(keys.duplicated().sum())
        missing_keys = int(keys.isna().any(axis=1).sum())
        if image_count != len(labels) or duplicate_keys or missing_keys:
            raise ValueError(f"Alignment failure for {current_year}: images={image_count}, labels={len(labels)}, duplicates={duplicate_keys}, missing_keys={missing_keys}")
        years.append({"year": current_year, "images": image_count, "labels": len(labels), "duplicate_keys": duplicate_keys, "missing_keys": missing_keys, "label_columns": list(labels.columns)})
    return {"data_dir": str(root), "horizon": horizon, "years": years, "total_images": sum(int(row["images"]) for row in years), "status": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--output", default="artifacts/data_audit_20d.json")
    args = parser.parse_args()
    result = audit(Path(args.data_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
