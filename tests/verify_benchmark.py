"""Fast structural checks for the repeatable benchmark runner."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from benchmark_gasket import (  # noqa: E402
    benchmark_size,
    expected_point_count,
    resolve_benchmark_devices,
    run_benchmarks,
)


def verify_expected_counts() -> None:
    assert expected_point_count(1) == 1
    assert expected_point_count(8) == 27
    assert expected_point_count(256) == 6561
    assert expected_point_count(7) is None


def verify_small_cpu_benchmark() -> None:
    result = benchmark_size(16, torch.device("cpu"), warmups=1, repeats=3)
    assert result.device == "cpu"
    assert result.rows == 16
    assert result.point_count == 81
    assert len(result.runs_ms) == 3
    assert result.minimum_ms > 0
    assert result.minimum_ms <= result.median_ms <= result.maximum_ms
    assert math.isclose(result.mean_ms, sum(result.runs_ms) / 3)
    assert result.candidate_megapixels_per_second > 0


def verify_result_order() -> None:
    results = run_benchmarks(
        [8, 16], [torch.device("cpu")], warmups=0, repeats=1
    )
    assert [(result.device, result.rows) for result in results] == [
        ("cpu", 8),
        ("cpu", 16),
    ]


def verify_device_expansion() -> None:
    devices = resolve_benchmark_devices(["cpu", "cpu"])
    assert devices == [torch.device("cpu")]


def verify_invalid_configuration() -> None:
    invalid_arguments = [
        (0, 0, 1),
        (8, -1, 1),
        (8, 0, 0),
    ]
    for rows, warmups, repeats in invalid_arguments:
        try:
            benchmark_size(rows, torch.device("cpu"), warmups, repeats)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid benchmark configuration was accepted")


def main() -> None:
    verify_expected_counts()
    verify_small_cpu_benchmark()
    verify_result_order()
    verify_device_expansion()
    verify_invalid_configuration()
    print("Benchmark verification passed")
    print("Warm-up/repeat handling, statistics, counts, and device order verified")


if __name__ == "__main__":
    main()

