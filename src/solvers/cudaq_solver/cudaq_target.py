"""CUDA-Q target selection for NVIDIA GPU execution."""

from __future__ import annotations

import cudaq


def ensure_cudaq_target() -> None:
    """Configure CUDA-Q to run on the NVIDIA backend.

    Raises:
        RuntimeError: If no NVIDIA GPU is available or the installed CUDA-Q build
            does not expose the ``nvidia`` target.
    """
    if cudaq.num_available_gpus() < 1:
        raise RuntimeError(
            "CUDA-Q requires an NVIDIA GPU for this backend, but no compatible GPU "
            "was detected. Use the Cirq or simulated annealing solver instead."
        )
    if not cudaq.has_target("nvidia"):
        raise RuntimeError(
            "CUDA-Q is installed, but the NVIDIA target is unavailable in this "
            "environment. Reinstall CUDA-Q with NVIDIA target support."
        )
    cudaq.set_target("nvidia", option="fp64")
