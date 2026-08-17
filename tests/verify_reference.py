"""Independent checks for the small NumPy/Python gasket reference."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from reference_gasket import centred_raster, pascal_odd_mask  # noqa: E402


def scalar_pascal_parity(rows: int) -> np.ndarray:
    """Compute the same mask independently using exact binomial integers."""

    expected = np.zeros((rows, rows), dtype=bool)
    for row in range(rows):
        for column in range(row + 1):
            expected[row, column] = math.comb(row, column) % 2 == 1
    return expected


def verify_known_rows() -> None:
    expected = [
        [1],
        [1, 1],
        [1, 0, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 0, 1],
    ]
    actual = pascal_odd_mask(len(expected))
    for row, values in enumerate(expected):
        assert actual[row, : row + 1].astype(int).tolist() == values


def verify_against_binomial_coefficients() -> None:
    actual = pascal_odd_mask(32)
    expected = scalar_pascal_parity(32)
    assert np.array_equal(actual, expected)


def verify_symmetry() -> None:
    mask = pascal_odd_mask(64)
    for row in range(mask.shape[0]):
        values = mask[row, : row + 1]
        assert np.array_equal(values, values[::-1])


def verify_power_of_two_counts() -> None:
    for exponent in range(7):
        rows = 2**exponent
        mask = pascal_odd_mask(rows)
        assert int(mask.sum()) == 3**exponent


def verify_raster_mapping() -> None:
    mask = pascal_odd_mask(32)
    raster = centred_raster(mask)
    assert raster.shape == (32, 63)
    assert raster.dtype == np.uint8
    assert int(raster.sum()) == int(mask.sum())


def main() -> None:
    verify_known_rows()
    verify_against_binomial_coefficients()
    verify_symmetry()
    verify_power_of_two_counts()
    verify_raster_mapping()
    print("Reference verification passed")
    print("Binary-address mask agrees with exact binomial coefficients")
    print("Row symmetry and 3^n point counts verified")
    print("Centred raster preserves every gasket point")


if __name__ == "__main__":
    main()

