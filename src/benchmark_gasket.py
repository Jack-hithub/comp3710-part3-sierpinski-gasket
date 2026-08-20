"""Repeatable CPU/MPS/CUDA benchmarks for PyTorch gasket generation."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from torch_gasket import generate_gasket, resolve_device


BENCHMARK_DEVICE_CHOICES = ("available", "cpu", "mps", "cuda")


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    rows: int
    warmups: int
    repeats: int
    point_count: int
    mean_ms: float
    median_ms: float
    stdev_ms: float
    minimum_ms: float
    maximum_ms: float
    runs_ms: tuple[float, ...]

    @property
    def candidate_megapixels_per_second(self) -> float:
        """Rate for the ``rows x rows`` candidate-address grid."""

        return self.rows**2 / (self.median_ms * 1_000)


def available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def resolve_benchmark_devices(requested: list[str]) -> list[torch.device]:
    """Resolve device names while preserving order and removing duplicates."""

    if not requested:
        raise ValueError("at least one device must be requested")

    expanded: list[torch.device] = []
    for name in requested:
        if name not in BENCHMARK_DEVICE_CHOICES:
            raise ValueError(f"unknown benchmark device: {name}")
        if name == "available":
            expanded.extend(available_devices())
        else:
            expanded.append(resolve_device(name))

    unique: list[torch.device] = []
    seen: set[str] = set()
    for device in expanded:
        if device.type not in seen:
            seen.add(device.type)
            unique.append(device)
    return unique


def expected_point_count(rows: int) -> int | None:
    """Return ``3^n`` when rows is ``2^n``; otherwise return ``None``."""

    if rows >= 1 and rows & (rows - 1) == 0:
        return 3 ** int(math.log2(rows))
    return None


def benchmark_size(
    rows: int,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> BenchmarkResult:
    if rows < 1:
        raise ValueError("rows must be at least 1")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    for _ in range(warmups):
        warmup_raster, _ = generate_gasket(rows, device)
        del warmup_raster

    timings: list[float] = []
    point_count = -1
    for _ in range(repeats):
        raster, elapsed_ms = generate_gasket(rows, device)
        point_count = int(raster.sum().item())
        timings.append(elapsed_ms)
        del raster

    expected = expected_point_count(rows)
    if expected is not None and point_count != expected:
        raise AssertionError(
            f"incorrect point count for {rows} rows: {point_count} != {expected}"
        )

    return BenchmarkResult(
        device=device.type,
        rows=rows,
        warmups=warmups,
        repeats=repeats,
        point_count=point_count,
        mean_ms=statistics.fmean(timings),
        median_ms=statistics.median(timings),
        stdev_ms=statistics.pstdev(timings),
        minimum_ms=min(timings),
        maximum_ms=max(timings),
        runs_ms=tuple(timings),
    )


def run_benchmarks(
    sizes: list[int],
    devices: list[torch.device],
    warmups: int,
    repeats: int,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for device in devices:
        for rows in sizes:
            results.append(benchmark_size(rows, device, warmups, repeats))
    return results


def save_results_csv(output_path: Path, results: list[BenchmarkResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "device",
                "rows",
                "warmups",
                "repeats",
                "point_count",
                "mean_ms",
                "median_ms",
                "stdev_ms",
                "minimum_ms",
                "maximum_ms",
                "candidate_megapixels_per_second",
                "runs_ms",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.device,
                    result.rows,
                    result.warmups,
                    result.repeats,
                    result.point_count,
                    f"{result.mean_ms:.9f}",
                    f"{result.median_ms:.9f}",
                    f"{result.stdev_ms:.9f}",
                    f"{result.minimum_ms:.9f}",
                    f"{result.maximum_ms:.9f}",
                    f"{result.candidate_megapixels_per_second:.9f}",
                    ";".join(f"{value:.9f}" for value in result.runs_ms),
                ]
            )


def save_benchmark_figure(
    output_path: Path, results: list[BenchmarkResult]
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)

    device_names = list(dict.fromkeys(result.device for result in results))
    for device_name in device_names:
        device_results = [
            result for result in results if result.device == device_name
        ]
        sizes = [result.rows for result in device_results]
        median_times = [result.median_ms for result in device_results]
        throughput = [
            result.candidate_megapixels_per_second for result in device_results
        ]
        axes[0].plot(sizes, median_times, marker="o", label=device_name)
        axes[1].plot(sizes, throughput, marker="o", label=device_name)

    axes[0].set_title("Median generation time")
    axes[0].set_xlabel("rows")
    axes[0].set_ylabel("milliseconds")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")

    axes[1].set_title("Candidate-grid throughput")
    axes[1].set_xlabel("rows")
    axes[1].set_ylabel("million candidate pixels / second")
    axes[1].set_xscale("log", base=2)

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(title="device")

    figure.suptitle("Sierpinski gasket PyTorch benchmark (warm-up excluded)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_results(results: list[BenchmarkResult]) -> None:
    print("device  rows  median_ms  mean_ms  stdev_ms  MPixels/s  points")
    for result in results:
        print(
            f"{result.device:<7} {result.rows:>5} "
            f"{result.median_ms:>10.3f} {result.mean_ms:>8.3f} "
            f"{result.stdev_ms:>9.3f} "
            f"{result.candidate_megapixels_per_second:>9.2f} "
            f"{result.point_count:>8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096]
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=BENCHMARK_DEVICE_CHOICES,
        default=["available"],
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--csv", type=Path, default=Path("outputs/benchmark_results.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmark.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_benchmark_devices(args.devices)
    results = run_benchmarks(args.sizes, devices, args.warmups, args.repeats)
    print_results(results)
    save_results_csv(args.csv, results)
    save_benchmark_figure(args.output, results)
    print(f"Saved results: {args.csv.resolve()}")
    print(f"Saved figure: {args.output.resolve()}")


if __name__ == "__main__":
    main()
