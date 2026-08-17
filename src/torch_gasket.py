"""PyTorch implementation of the Sierpinski gasket binary-address test."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit or automatic PyTorch device selection."""

    requested = requested.lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def synchronise(device: torch.device) -> None:
    """Wait for queued accelerator work before recording a timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def pascal_odd_mask_torch(rows: int, device: torch.device) -> torch.Tensor:
    """Return the gasket as a triangular Boolean tensor on ``device``.

    Broadcasting creates virtual row and column grids with shapes ``(rows, 1)``
    and ``(1, rows)``. Every output position is then evaluated by the same
    binary-address expression in parallel.
    """

    if rows < 1:
        raise ValueError("rows must be at least 1")

    addresses = torch.arange(rows, dtype=torch.int32, device=device)
    row = addresses[:, None]
    column = addresses[None, :]
    inside_triangle = column <= row
    odd = torch.bitwise_and(column, row - column) == 0
    return inside_triangle & odd


def centred_raster_torch(mask: torch.Tensor) -> torch.Tensor:
    """Map a square triangular mask to a centred ``rows x (2*rows-1)`` raster."""

    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be a square 2D tensor")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have torch.bool dtype")

    rows = mask.shape[0]
    raster = torch.zeros(
        (rows, 2 * rows - 1), dtype=torch.uint8, device=mask.device
    )
    y, x = torch.nonzero(mask, as_tuple=True)
    display_x = (rows - 1 - y) + 2 * x
    raster[y, display_x] = 1
    return raster


def generate_gasket(rows: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """Generate a centred raster and return its elapsed device time in ms."""

    synchronise(device)
    start = time.perf_counter()
    mask = pascal_odd_mask_torch(rows, device)
    raster = centred_raster_torch(mask)
    synchronise(device)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return raster, elapsed_ms


def save_figure(
    raster: torch.Tensor,
    output_path: Path,
    device: torch.device,
    elapsed_ms: float,
) -> None:
    """Transfer one finished raster to CPU and save a labelled figure."""

    import matplotlib.pyplot as plt

    image_array: np.ndarray = raster.cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    image = axis.imshow(
        image_array,
        cmap="viridis",
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    axis.set_title(
        f"PyTorch Sierpinski gasket - device={device.type}, "
        f"generation={elapsed_ms:.3f} ms"
    )
    axis.set_xlabel("centred horizontal pixel")
    axis.set_ylabel("Pascal row")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, ticks=[0, 1], label="entry parity")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/torch_gasket.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    raster, elapsed_ms = generate_gasket(args.rows, device)
    point_count = int(raster.sum().item())
    save_figure(raster, args.output, device, elapsed_ms)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device.type}")
    print(f"Rows: {args.rows}")
    print(f"Odd Pascal entries: {point_count}")
    print(f"Raster shape: {tuple(raster.shape)}")
    print(f"Generation time: {elapsed_ms:.3f} ms")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

