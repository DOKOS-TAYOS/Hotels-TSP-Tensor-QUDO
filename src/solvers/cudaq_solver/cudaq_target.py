"""CUDA-Q target selection for GPU and density-matrix backends.

Strategy
--------
1. **Noiseless** — use the ``nvidia`` GPU target (fp64).
2. **Noisy + GPU trajectory support** — stay on ``nvidia`` (fp64).  CUDA-Q
   ≥ 0.7 supports passing a ``noise_model`` to ``cudaq.sample()`` on the
   ``nvidia`` target, which uses trajectory-based simulation (O(2ⁿ) memory,
   GPU-accelerated).
3. **Noisy + no trajectory support** — fall back to ``density-matrix-cpu``
   (O(4ⁿ) memory, CPU only).

The GPU probe is executed **once** per process and cached.
"""

from __future__ import annotations

import logging

import cudaq

from solvers.noise import NoiseConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (reset via ``reset_target_state()`` for tests)
# ---------------------------------------------------------------------------
_gpu_noise_support: bool | None = None   # None = not yet probed
_current_target: str | None = None       # last target set by us


def _gpu_supports_noise() -> bool:
    """Probe whether the ``nvidia`` target accepts ``noise_model`` in ``sample``.

    Runs a trivial one-qubit circuit with depolarizing noise. On failure,
    callers should use ``density-matrix-cpu``. Result is cached per process.
    Temporarily switches the global CUDA-Q target during the probe.

    Returns:
        True if GPU trajectory noise works; False if probe fails or no GPU.
    """
    global _gpu_noise_support
    if _gpu_noise_support is not None:
        return _gpu_noise_support

    try:
        if cudaq.num_available_gpus() < 1 or not cudaq.has_target("nvidia"):
            _gpu_noise_support = False
            return False

        cudaq.set_target("nvidia", option="fp64")

        @cudaq.kernel
        def _probe_kernel():
            q = cudaq.qvector(1)
            x(q[0])  # noqa: F821

        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.DepolarizationChannel(0.01))

        cudaq.sample(_probe_kernel, shots_count=4, noise_model=noise)

        logger.info(
            "GPU trajectory noise probe succeeded — noisy simulations "
            "will use the 'nvidia' GPU target."
        )
        _gpu_noise_support = True
    except Exception:  # noqa: BLE001
        logger.info(
            "GPU trajectory noise probe failed — noisy simulations "
            "will fall back to 'density-matrix-cpu'."
        )
        _gpu_noise_support = False

    return _gpu_noise_support


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_cudaq_target(noise_config: NoiseConfig | None = None) -> str:
    """Configure CUDA-Q to run on the appropriate backend.

    Target selection matrix:

    +------------------+-----------------+--------------------------+
    | noise enabled?   | GPU + trajectory| Target chosen            |
    +==================+=================+==========================+
    | No               | GPU available   | ``nvidia`` (fp64)        |
    +------------------+-----------------+--------------------------+
    | No               | no GPU          | **RuntimeError**         |
    +------------------+-----------------+--------------------------+
    | Yes              | probe OK        | ``nvidia`` (fp64, traj.) |
    +------------------+-----------------+--------------------------+
    | Yes              | probe fail      | ``density-matrix-cpu``   |
    +------------------+-----------------+--------------------------+

    Idempotent: if the desired target is already active, ``set_target`` is
    **not** called again.

    Args:
        noise_config: Optional noise parameters.  ``None`` ≡ noise disabled.

    Returns:
        The name of the active target (``"nvidia"`` or
        ``"density-matrix-cpu"``).

    Raises:
        RuntimeError: If no NVIDIA GPU is available (noiseless mode) or the
            required target is missing in this CUDA-Q build.
    """
    global _current_target

    noise_enabled = noise_config is not None and noise_config.enabled

    if noise_enabled:
        if _gpu_supports_noise():
            target_name = "nvidia"
        else:
            # Fall back to CPU density-matrix.
            if not cudaq.has_target("density-matrix-cpu"):
                raise RuntimeError(
                    "CUDA-Q noise simulation requires GPU trajectory support or "
                    "the 'density-matrix-cpu' target, but neither is available."
                )
            target_name = "density-matrix-cpu"
    else:
        # Noiseless GPU path (original behaviour).
        if cudaq.num_available_gpus() < 1:
            raise RuntimeError(
                "CUDA-Q requires an NVIDIA GPU for this backend, but no "
                "compatible GPU was detected. Use the Cirq or simulated "
                "annealing solver instead."
            )
        if not cudaq.has_target("nvidia"):
            raise RuntimeError(
                "CUDA-Q is installed, but the NVIDIA target is unavailable "
                "in this environment. Reinstall CUDA-Q with NVIDIA target "
                "support."
            )
        target_name = "nvidia"

    # Idempotent: skip if already on the right target.
    if _current_target != target_name:
        if target_name == "nvidia":
            cudaq.set_target("nvidia", option="fp64")
        else:
            cudaq.set_target(target_name)
        _current_target = target_name
        logger.debug("CUDA-Q target set to '%s'.", target_name)

    return target_name


def get_current_target() -> str | None:
    """Return the name of the target last set by :func:`ensure_cudaq_target`."""
    return _current_target


def reset_target_state() -> None:
    """Reset the module-level cache (useful in tests).

    After calling this, the next :func:`ensure_cudaq_target` invocation will
    re-run the GPU noise probe and set the target unconditionally.
    """
    global _gpu_noise_support, _current_target
    _gpu_noise_support = None
    _current_target = None
