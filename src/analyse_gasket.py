"""Box-counting and binary-address visual analysis of the gasket."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from reference_gasket import centred_raster
from torch_gasket import DEVICE_CHOICES, pascal_odd_mask_torch, resolve_device, synchronise


def count_occupied_boxes(mask: np.ndarray, box_size: int) -> int:
    """Count square boxes containing at least one foreground pixel."""

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if box_size < 1:
        raise ValueError("box_size must be at least 1")

    height, width = mask.shape
    padded_height = math.ceil(height / box_size) * box_size
    padded_width = math.ceil(width / box_size) * box_size
    padded = np.zeros((padded_height, padded_width), dtype=bool)
    padded[:height, :width] = mask.astype(bool, copy=False)

    blocks = padded.reshape(
        padded_height // box_size,
        box_size,
        padded_width // box_size,
        box_size,
    )
    occupied = blocks.any(axis=(1, 3))
    return int(occupied.sum())


def power_of_two_box_sizes(rows: int) -> list[int]:
    """Return aligned scales from one pixel through one quarter of the image."""

    if rows < 4 or rows & (rows - 1):
        raise ValueError("rows must be a power of two and at least 4")
    return [2**exponent for exponent in range(int(math.log2(rows)) - 1)]


def estimate_box_dimension(
    mask: np.ndarray, box_sizes: list[int]
) -> tuple[np.ndarray, float, float]:
    """Return box counts, fitted dimension, and log-space R-squared."""

    if len(box_sizes) < 2:
        raise ValueError("at least two box sizes are required")

    counts = np.asarray(
        [count_occupied_boxes(mask, size) for size in box_sizes],
        dtype=np.float64,
    )
    inverse_scale = mask.shape[0] / np.asarray(box_sizes, dtype=np.float64)
    x = np.log(inverse_scale)
    y = np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    return counts.astype(np.int64), float(slope), r_squared


def bit_counts(values: np.ndarray, number_of_bits: int) -> np.ndarray:
    """Count set bits using vector operations and a loop over bit positions."""

    work = values.astype(np.int64, copy=True)
    counts = np.zeros_like(work, dtype=np.int16)
    for _ in range(number_of_bits):
        counts += (work & 1).astype(np.int16)
        work >>= 1
    return counts


def binary_address_colour(mask: np.ndarray) -> np.ndarray:
    """Colour points by the balance of left and right binary address bits."""

    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be a square 2D array")

    rows = mask.shape[0]
    number_of_bits = max(1, (rows - 1).bit_length())
    y, x = np.nonzero(mask)
    other_address = y - x
    balance = bit_counts(x, number_of_bits) - bit_counts(
        other_address, number_of_bits
    )

    colour_raster = np.full((rows, 2 * rows - 1), np.nan, dtype=np.float32)
    display_x = (rows - 1 - y) + 2 * x
    colour_raster[y, display_x] = balance
    return colour_raster


def save_box_counts(
    output_path: Path, box_sizes: list[int], counts: np.ndarray
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["box_size", "occupied_boxes"])
        writer.writerows(zip(box_sizes, counts.tolist()))


def save_analysis_figure(
    mask: np.ndarray,
    box_sizes: list[int],
    counts: np.ndarray,
    estimated_dimension: float,
    r_squared: float,
    theoretical_dimension: float,
    device: torch.device,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_raster = centred_raster(mask)
    colour_raster = binary_address_colour(mask)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)

    axes[0].imshow(binary_raster, cmap="gray", interpolation="nearest", aspect="equal")
    axes[0].set_title("Binary parity view")

    address_cmap = plt.get_cmap("coolwarm").copy()
    address_cmap.set_bad("black")
    colour_limit = float(np.nanmax(np.abs(colour_raster)))
    address_image = axes[1].imshow(
        colour_raster,
        cmap=address_cmap,
        vmin=-colour_limit,
        vmax=colour_limit,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Binary address balance")
    figure.colorbar(
        address_image,
        ax=axes[1],
        label="set bits in c minus set bits in (row-c)",
    )

    inverse_scale = mask.shape[0] / np.asarray(box_sizes, dtype=np.float64)
    x = np.log(inverse_scale)
    y = np.log(counts)
    fitted = estimated_dimension * x + np.polyfit(x, y, 1)[1]
    axes[2].scatter(x, y, color="tab:blue", label="measured box counts")
    axes[2].plot(
        x,
        fitted,
        color="tab:orange",
        label=f"fit: D={estimated_dimension:.6f}",
    )
    axes[2].set_title("Box-counting dimension")
    axes[2].set_xlabel("log(image width / box size)")
    axes[2].set_ylabel("log(occupied boxes)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    for axis in axes[:2]:
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        f"Sierpinski gasket analysis - device={device.type}, "
        f"theory={theoretical_dimension:.6f}, R^2={r_squared:.6f}",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gasket_analysis.png")
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("outputs/box_counts.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    box_sizes = power_of_two_box_sizes(args.rows)

    mask_tensor = pascal_odd_mask_torch(args.rows, device)
    synchronise(device)
    mask = mask_tensor.cpu().numpy()
    counts, estimate, r_squared = estimate_box_dimension(mask, box_sizes)
    theoretical = math.log(3) / math.log(2)

    save_box_counts(args.csv, box_sizes, counts)
    save_analysis_figure(
        mask,
        box_sizes,
        counts,
        estimate,
        r_squared,
        theoretical,
        device,
        args.output,
    )

    print(f"Device: {device.type}")
    print(f"Rows: {args.rows}")
    print("Box size -> occupied boxes")
    for size, count in zip(box_sizes, counts):
        print(f"{size:>8} -> {int(count)}")
    print(f"Estimated dimension: {estimate:.9f}")
    print(f"Theoretical dimension: {theoretical:.9f}")
    print(f"Absolute error: {abs(estimate - theoretical):.3e}")
    print(f"R-squared: {r_squared:.9f}")
    print(f"Saved figure: {args.output.resolve()}")
    print(f"Saved counts: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
