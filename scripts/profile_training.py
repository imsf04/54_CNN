"""Measure data-path and step throughput of the 20-day CNN on the full training split."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.data import IterableImageDataset, load_real_data, temporal_split
from price_trends.models import PriceTrendCNN

MEAN = 0.10689710380321306
STD = 0.3089823829956947


def make_loader(bundle, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = IterableImageDataset(bundle, batch_size=batch_size, mean=MEAN, std=STD, shuffle=shuffle, seed=7)
    return DataLoader(dataset, batch_size=None, pin_memory=True)


class UtilSampler:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> list[int]:
        self.process.terminate()
        out, _ = self.process.communicate()
        return [int(value) for value in out.split() if value.strip().isdigit()]


def loader_throughput(loader: DataLoader, batches: int) -> float:
    stream = iter(loader)
    count = 0
    start = time.perf_counter()
    for _ in range(batches):
        x, _ = next(stream)
        count += len(x)
    return count / (time.perf_counter() - start)


def train_steps(
    model,
    loader: DataLoader,
    device: str,
    steps: int,
    warmup: int,
    use_amp: bool,
    channels_last: bool = False,
) -> dict[str, float]:
    def move_image(batch):
        if channels_last:
            return batch.to(device, non_blocking=True, memory_format=torch.channels_last)
        return batch.to(device, non_blocking=True)

    def move_label(batch):
        return batch.to(device, non_blocking=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stream = iter(loader)
    for _ in range(warmup):
        x, y = next(stream)
        x, y = move_image(x), move_label(y)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    sampler = UtilSampler()
    waits: list[float] = []
    count = 0
    previous = time.perf_counter()
    start = previous
    for _ in range(steps):
        x, y = next(stream)
        loaded = time.perf_counter()
        waits.append(loaded - previous)
        x, y = move_image(x), move_label(y)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        previous = time.perf_counter()
        count += len(y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    samples = sampler.stop()
    return {
        "samples_per_second": count / elapsed,
        "ms_per_step": elapsed / steps * 1000,
        "loader_wait_median_ms": statistics.median(waits) * 1000,
        "gpu_util_median": statistics.median(samples) if samples else -1,
        "gpu_util_mean": statistics.fmean(samples) if samples else -1,
        "gpu_util_p10": sorted(samples)[len(samples) // 10] if samples else -1,
    }


@torch.no_grad()
def eval_throughput(model, loader: DataLoader, device: str, use_amp: bool) -> dict[str, float]:
    model.eval()
    count = 0
    start = time.perf_counter()
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            model(x.to(device, non_blocking=True))
        count += len(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {"samples": count, "seconds": elapsed, "samples_per_second": count / elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/extracted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    bundle = load_real_data(args.data_dir, target="Retx_20d_label", horizon="20d", return_column="Retx_20d", lazy=True)
    splits = temporal_split(bundle)
    train_loader = make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(splits["val"], args.batch_size, shuffle=False)

    report = {
        "loader": {"samples_per_second": loader_throughput(make_loader(splits["train"], args.batch_size, True), 40)}
    }
    model = PriceTrendCNN().to(args.device)
    report["baseline"] = train_steps(model, train_loader, args.device, args.steps, args.warmup, True)
    print(json.dumps({"baseline": report["baseline"]}), flush=True)

    channels_last = PriceTrendCNN().to(args.device, memory_format=torch.channels_last)
    report["channels_last"] = train_steps(
        channels_last, train_loader, args.device, args.steps, args.warmup, True, channels_last=True
    )
    print(json.dumps({"channels_last": report["channels_last"]}), flush=True)

    compiled = torch.compile(PriceTrendCNN().to(args.device), mode="reduce-overhead")
    report["compile_reduce_overhead"] = train_steps(
        compiled, train_loader, args.device, args.steps, args.warmup, True
    )
    print(json.dumps({"compile_reduce_overhead": report["compile_reduce_overhead"]}), flush=True)

    for size in (256, 512, 1024):
        wide = make_loader(splits["train"], size, shuffle=True)
        burst = train_steps(PriceTrendCNN().to(args.device), wide, args.device, 25, 5, True)
        report[f"batch_{size}"] = burst
        print(json.dumps({f"batch_{size}": burst}), flush=True)

    report["eval_fp32_batch128"] = eval_throughput(model, val_loader, args.device, False)
    report["eval_amp_batch128"] = eval_throughput(model, val_loader, args.device, True)
    val_1024 = make_loader(splits["val"], 1024, shuffle=False)
    report["eval_fp32_batch1024"] = eval_throughput(model, val_1024, args.device, False)
    for name in ("eval_fp32_batch128", "eval_amp_batch128", "eval_fp32_batch1024"):
        print(json.dumps({name: report[name]}), flush=True)


if __name__ == "__main__":
    main()
