from __future__ import annotations
from pathlib import Path
from urllib.request import urlopen
import zipfile

DEFAULT_URLS = {"20d": "https://dachxiu.chicagobooth.edu/download/img_data.zip"}

def download_and_extract(horizon, output_dir, url=None, timeout=60):
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"img_data_{horizon}.zip"; source = url or DEFAULT_URLS.get(horizon)
    if not source:
        raise ValueError(f"No verified default URL is recorded for {horizon}; pass --url or use a local data directory")
    with urlopen(source, timeout=timeout) as response, archive.open("wb") as handle:
        handle.write(response.read())
    with zipfile.ZipFile(archive) as zf: zf.extractall(destination)
    return destination
