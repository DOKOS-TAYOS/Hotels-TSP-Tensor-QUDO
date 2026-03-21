"""Build CUDA-Q noise models from a :class:`NoiseConfig`.

Translates the backend-agnostic :class:`NoiseConfig` into a
``cudaq.NoiseModel`` that can be passed to ``cudaq.sample(…, noise_model=…)``.

Design notes
------------
* ``build_noise_model`` adds noise to **all** single-qubit gates using
  ``add_all_qubit_channel``.  Per-gate overrides from
  ``NoiseConfig.gate_noise`` are applied first; gates not listed there fall
  back to the default ``NoiseConfig.probability``.
* The ``nvidia`` GPU target (CUDA-Q ≥ 0.7) supports **trajectory-based**
  noise simulation with O(2ⁿ) memory and GPU acceleration.  When this is
  available the target stays on ``nvidia``.  Otherwise CUDA-Q falls back to
  ``density-matrix-cpu`` (CPU-only, O(4ⁿ) memory).  Target selection is
  handled by :mod:`solvers.cudaq_solver.cudaq_target`
  (see :func:`ensure_cudaq_target`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cudaq

from solvers.noise import NoiseConfig

if TYPE_CHECKING:
    pass


# Gate names that typically receive single-qubit noise.
# You can extend this list to cover more gates as needed.
_DEFAULT_SINGLE_QUBIT_GATES: tuple[str, ...] = (
    "h",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "r1",
    "s",
    "t",
)

# Gate names for two-qubit noise (CNOT, CZ, etc.).
_DEFAULT_TWO_QUBIT_GATES: tuple[str, ...] = (
    "cx",
    "cz",
    "swap",
)


def _make_channel(noise_type: str, probability: float) -> cudaq.KrausChannel:
    """Create a single CUDA-Q noise channel from a type name and probability.

    Raises:
        ValueError: If ``noise_type`` is not recognized.
    """
    factories: dict[str, type] = {
        "depolarizing": cudaq.DepolarizationChannel,
        "amplitude_damping": cudaq.AmplitudeDampingChannel,
        "phase_damping": cudaq.PhaseDampingChannel,
        "bit_flip": cudaq.BitFlipChannel,
        "phase_flip": cudaq.PhaseFlipChannel,
    }
    factory = factories.get(noise_type)
    if factory is None:
        raise ValueError(f"Unknown noise type for CUDA-Q: {noise_type!r}")
    return factory(probability)


def build_noise_model(config: NoiseConfig) -> cudaq.NoiseModel:
    """Construct a ``cudaq.NoiseModel`` from a :class:`NoiseConfig`.

    Args:
        config: Backend-agnostic noise configuration.

    Returns:
        A ``cudaq.NoiseModel`` ready to be passed as
        ``cudaq.sample(kernel, …, noise_model=noise)``.
    """
    noise = cudaq.NoiseModel()

    # Apply per-gate overrides first, then the default to remaining gates.
    overridden_gates: set[str] = set()
    for gate_name, gate_prob in config.gate_noise.items():
        channel = _make_channel(config.noise_type, gate_prob)
        noise.add_all_qubit_channel(gate_name, channel)
        overridden_gates.add(gate_name)

    # Default probability for all single-qubit gates not explicitly overridden.
    for gate_name in _DEFAULT_SINGLE_QUBIT_GATES:
        if gate_name not in overridden_gates:
            channel = _make_channel(config.noise_type, config.probability)
            noise.add_all_qubit_channel(gate_name, channel)

    # Two-qubit gate noise: only depolarizing is applied.  Non-depolarizing
    # channels (amplitude_damping, phase_damping, bit_flip, phase_flip) are
    # inherently single-qubit and have no standard two-qubit generalisation,
    # so they are intentionally skipped here.  Note: this differs from the
    # Cirq qubit backend, which applies the single-qubit channel independently
    # to each qubit after every gate (including multi-qubit gates).
    # See the NoiseConfig docstring in solvers/noise.py for the full
    # cross-backend comparison.
    if config.noise_type == "depolarizing":
        for gate_name in _DEFAULT_TWO_QUBIT_GATES:
            if gate_name not in overridden_gates:
                two_q_channel = cudaq.DepolarizationChannel(config.probability)
                noise.add_all_qubit_channel(gate_name, two_q_channel, num_controls=1)

    return noise


def get_noise_model(
    config: NoiseConfig | None,
) -> cudaq.NoiseModel | None:
    """Return a ``cudaq.NoiseModel`` when noise is enabled, else ``None``.

    This is the main entry point for CUDA-Q circuit modules.  When the
    returned value is not ``None``, callers must pass
    ``noise_model=…`` to every ``cudaq.sample`` call.  Target selection
    (GPU trajectory vs. ``density-matrix-cpu``) is handled by
    :func:`ensure_cudaq_target`.
    """
    if config is None or not config.enabled:
        return None
    return build_noise_model(config)
