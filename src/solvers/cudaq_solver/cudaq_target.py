"""CUDA-Q target selection (GPU vs CPU simulator)."""

from __future__ import annotations

import cudaq


def ensure_cudaq_target() -> None:
    """Set CUDA-Q target to GPU if available, else CPU simulator."""
    if cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia"):
        cudaq.set_target("nvidia", option="fp64")
    else:
        print(
            "CUDA or GPU support is unavailable. Running with CPU simulator. "
            "Performance may be significantly reduced."
        )
        cudaq.set_target("qpp-cpu")
