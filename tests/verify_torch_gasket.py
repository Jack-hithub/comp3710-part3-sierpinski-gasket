"""Compare the PyTorch gasket with the independent NumPy reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from reference_gasket import centred_raster, pascal_odd_mask  # noqa: E402
from torch_gasket import (  # noqa: E402
    centred_raster_torch,
    pascal_odd_mask_torch,
    resolve_device,
)


def verify_device_selection() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
    try:
        resolve_device("not-a-device")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid device name was accepted")


def verify_against_reference(device: torch.device) -> None:
    for rows in (1, 2, 3, 8, 31, 64, 128):
        expected_mask = pascal_odd_mask(rows)
        actual_mask = pascal_odd_mask_torch(rows, device)
        assert actual_mask.dtype == torch.bool
        assert actual_mask.device.type == device.type
        assert np.array_equal(actual_mask.cpu().numpy(), expected_mask)

        expected_raster = centred_raster(expected_mask)
        actual_raster = centred_raster_torch(actual_mask)
        assert actual_raster.dtype == torch.uint8
        assert np.array_equal(actual_raster.cpu().numpy(), expected_raster)


def verify_known_counts(device: torch.device) -> None:
    for exponent in range(8):
        rows = 2**exponent
        mask = pascal_odd_mask_torch(rows, device)
        assert int(mask.sum().item()) == 3**exponent


def verify_invalid_inputs() -> None:
    try:
        pascal_odd_mask_torch(0, torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("zero rows were accepted")

    wrong_dtype = torch.zeros((4, 4), dtype=torch.int32)
    try:
        centred_raster_torch(wrong_dtype)
    except TypeError:
        pass
    else:
        raise AssertionError("non-Boolean mask was accepted")


def available_test_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def main() -> None:
    verify_device_selection()
    verify_invalid_inputs()
    devices = available_test_devices()
    for device in devices:
        verify_against_reference(device)
        verify_known_counts(device)
        print(f"Verified device: {device.type}")

    print("PyTorch verification passed")
    print("PyTorch masks and centred rasters agree with the NumPy reference")
    print("Boolean dtype, device placement, input checks, and 3^n counts passed")


if __name__ == "__main__":
    main()

