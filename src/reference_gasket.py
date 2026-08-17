"""Small independent reference implementation of the Sierpinski gasket.

The construction uses the parity pattern in Pascal's triangle.  Entry
``(row, column)`` is odd exactly when the binary representations of ``column``
and ``row - column`` have no 1-bit in the same position.  This is the binary
address condition described in Section 2.10 of *Fractals for the Classroom*.

This NumPy version is intentionally kept separate from the later PyTorch
implementation so it can serve as a numerical reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pascal_odd_mask(rows: int) -> np.ndarray:
    """Return a triangular Boolean mask for odd Pascal-triangle entries.

    ``mask[row, column]`` is true when ``0 <= column <= row`` and the
    corresponding binomial coefficient is odd.
    """

    if rows < 1:
        raise ValueError("rows must be at least 1")

    row = np.arange(rows, dtype=np.int64)[:, None]
    column = np.arange(rows, dtype=np.int64)[None, :]
    inside_triangle = column <= row

    # Lucas' theorem modulo 2 gives this compact binary-address test.
    odd = (column & (row - column)) == 0
    return inside_triangle & odd


def centred_raster(mask: np.ndarray) -> np.ndarray:
    """Place a triangular Pascal mask in a centred display raster."""

    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be a square 2D array")

    rows = mask.shape[0]
    raster = np.zeros((rows, 2 * rows - 1), dtype=np.uint8)
    y, x = np.nonzero(mask)
    display_x = (rows - 1 - y) + 2 * x
    raster[y, display_x] = 1
    return raster


def save_reference_figure(raster: np.ndarray, output_path: Path) -> None:
    """Save a labelled, nearest-neighbour view of a gasket raster."""

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = axis.imshow(
        raster,
        cmap="magma",
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    axis.set_title("Sierpinski gasket reference - odd entries of Pascal's triangle")
    axis.set_xlabel("centred horizontal pixel")
    axis.set_ylabel("Pascal row")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, ticks=[0, 1], label="entry parity (0 even, 1 odd)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reference_gasket.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask = pascal_odd_mask(args.rows)
    raster = centred_raster(mask)
    save_reference_figure(raster, args.output)
    print(f"Rows: {args.rows}")
    print(f"Odd Pascal entries: {int(mask.sum())}")
    print(f"Raster shape: {raster.shape}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

