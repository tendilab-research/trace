"""The **single** point of device resolution.

Other modules never use the literal strings 'cuda' / 'mps' / 'cpu'. They must
go through `resolve_device()`. Reasons:

  - Local (macOS/arm64) is MPS, the GPU pod (A100) is CUDA. If the two backends
    branch throughout the code, gate 7 (reproducibility) failures cannot be localized.
  - MPS does not support float64. Only this module should know that
    (`torch_dtype`).
  - Every output row must carry `device_info()` as provenance so the CPU/GPU
    cross-check (gate 7) can tell after the fact which backend produced it.
"""

from __future__ import annotations

import functools
import os
import platform

VALID = ("cuda", "mps", "cpu")


class DeviceError(RuntimeError):
    """Requested device is unavailable. Prevents silent CPU fallback."""


MPS_MIN_TORCH = (2, 6)      # minimum torch version for which TabPFN accepts MPS


def _available(name: str) -> bool:
    import torch

    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        # Even if torch supports MPS, TabPFN refuses it:
        #   "The MPS device was selected, but TabPFN requires PyTorch >= 2.6 for MPS."
        # Without filtering here, autodetect picks mps and the first fit crashes.
        v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        return torch.backends.mps.is_available() and v >= MPS_MIN_TORCH
    return True


@functools.lru_cache(maxsize=None)
def resolve_device(prefer: str | None = None) -> str:
    """Determine the device to use.

    Priority: `prefer` argument > env var ``TRACE_DEVICE`` > autodetect (cuda>mps>cpu).

    If an explicitly requested device is unavailable, **raise**. A silent CPU
    fallback would mean "rented six A100s and ran on CPU". Fallback happens
    only under autodetect.
    """
    want = prefer or os.environ.get("TRACE_DEVICE") or ""
    want = want.strip().lower()

    if want:
        if want not in VALID:
            raise DeviceError(f"TRACE_DEVICE={want!r}; allowed values are {VALID}")
        if not _available(want):
            raise DeviceError(
                f"Device {want!r} was requested but is unavailable. "
                f"Autodetect result is {_autodetect()!r}."
            )
        return want

    return _autodetect()


def _autodetect() -> str:
    for name in VALID:
        if _available(name):
            return name
    return "cpu"


def torch_dtype(device: str | None = None):
    """Floating-point dtype for the given backend.

    MPS has no float64 kernels (``TypeError: Cannot convert a MPS Tensor to
    float64``). CUDA/CPU use float64.

    At the current pin (torch 2.5.1) TabPFN refuses MPS, so local runs on CPU
    -> local and pod are **both float64**. Any gate 7b difference comes only
    from CUDA kernel nondeterminism, not dtype.
    """
    import torch

    dev = device or resolve_device()
    return torch.float32 if dev == "mps" else torch.float64


def supports_float64(device: str | None = None) -> bool:
    return (device or resolve_device()) != "mps"


def device_info(device: str | None = None) -> dict:
    """Dict to store as-is in the provenance columns of the output parquet."""
    import torch

    dev = device or resolve_device()
    info = {
        "device": dev,
        "torch": torch.__version__,
        "platform": f"{platform.system()}-{platform.machine()}",
        "float64": supports_float64(dev),
    }
    if dev == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda"] = torch.version.cuda
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(device_info(), indent=2, ensure_ascii=False))
