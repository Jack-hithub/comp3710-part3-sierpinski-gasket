"""Checks for box counting and binary-address colour analysis."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from analyse_gasket import (  # noqa: E402
    binary_address_colour,
    count_occupied_boxes,
    estimate_box_dimension,
    power_of_two_box_sizes,
)
from torch_gasket import pascal_odd_mask_torch  # noqa: E402


def verify_aligned_box_counts() -> None:
    exponent = 8
    rows = 2**exponent
    mask = pascal_odd_mask_torch(rows, device="cpu").numpy()
    box_sizes = power_of_two_box_sizes(rows)
    counts, estimate, r_squared = estimate_box_dimension(mask, box_sizes)

    expected = np.asarray(
        [3 ** (exponent - int(math.log2(size))) for size in box_sizes]
    )
    assert np.array_equal(counts, expected)
    assert abs(estimate - math.log(3) / math.log(2)) < 1e-12
    assert abs(r_squared - 1.0) < 1e-12


def verify_padding_for_arbitrary_shape() -> None:
    mask = np.zeros((3, 5), dtype=bool)
    mask[0, 0] = True
    mask[2, 4] = True
    assert count_occupied_boxes(mask, 2) == 2
    assert count_occupied_boxes(mask, 4) == 2


def verify_address_colour() -> None:
    rows = 64
    mask = pascal_odd_mask_torch(rows, device="cpu").numpy()
    colour = binary_address_colour(mask)
    assert colour.shape == (rows, 2 * rows - 1)
    assert int(np.isfinite(colour).sum()) == int(mask.sum())
    finite = colour[np.isfinite(colour)]
    assert finite.min() < 0
    assert finite.max() > 0


def verify_invalid_inputs() -> None:
    for rows in (0, 3, 4, 6):
        try:
            power_of_two_box_sizes(rows)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid row count accepted: {rows}")


def main() -> None:
    verify_aligned_box_counts()
    verify_padding_for_arbitrary_shape()
    verify_address_colour()
    verify_invalid_inputs()
    print("Analysis verification passed")
    print("Aligned box counts follow the exact 3^n scaling law")
    print("Estimated dimension agrees with log(3) / log(2)")
    print("Address colours preserve exactly the foreground gasket points")


if __name__ == "__main__":
    main()
