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

import cudaq

from solvers.noise import NoiseConfig


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

# Depolarizing after controlled gates: CUDA-Q expects the *base* op name plus
# ``num_controls``, not composite names like ``cx`` / ``cz`` (those raise
# ``Invalid quantum op for noise_model::add_channel`` on current targets).
_DEFAULT_TWO_QUBIT_NOISE: tuple[tuple[str, int], ...] = (
    ("x", 1),  # CNOT, ``x.ctrl``, ``cx`` in kernels
    ("z", 1),  # CZ, ``z.ctrl``
)

# Map :class:`NoiseConfig` / YAML ``gate_noise`` keys to CUDA-Q (op, num_controls).
_TWO_QUBIT_GATE_NOISE_ALIASES: dict[str, tuple[str, int]] = {
    "cx": ("x", 1),
    "cnot": ("x", 1),
    "cz": ("z", 1),
    "swap": ("swap", 0),
}


def _make_channel(noise_type: str, probability: float) -> cudaq.KrausChannel:
    """Create a single CUDA-Q noise channel from a type name and probability.

    Channel constructors are resolved lazily so one missing symbol (e.g. across
    CUDA-Q versions) does not break unrelated noise types.  Phase dephasing is
    ``cudaq.PhaseDamping`` in current releases; older builds used
    ``PhaseDampingChannel`` when present.

    Raises:
        ValueError: If ``noise_type`` is not recognized.
    """
    if noise_type == "depolarizing":
        return cudaq.DepolarizationChannel(probability)
    if noise_type == "amplitude_damping":
        return cudaq.AmplitudeDampingChannel(probability)
    if noise_type == "phase_damping":
        legacy = getattr(cudaq, "PhaseDampingChannel", None)
        if legacy is not None:
            return legacy(probability)
        return cudaq.PhaseDamping(probability)
    if noise_type == "bit_flip":
        return cudaq.BitFlipChannel(probability)
    if noise_type == "phase_flip":
        return cudaq.PhaseFlipChannel(probability)
    raise ValueError(f"Unknown noise type for CUDA-Q: {noise_type!r}")


def _two_qubit_depolarizing_channel(probability: float) -> cudaq.KrausChannel | None:
    """2-qubit depolarizing Kraus ops for controlled gates (dim 4).  Returns ``None`` if unsupported."""
    factory = getattr(cudaq, "Depolarization2", None)
    if factory is None:
        return None
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
    overridden_single: set[str] = set()
    overridden_two_qubit: set[tuple[str, int]] = set()
    for gate_name, gate_prob in config.gate_noise.items():
        alias = _TWO_QUBIT_GATE_NOISE_ALIASES.get(gate_name)
        if alias is not None:
            op, num_controls = alias
            if config.noise_type != "depolarizing":
                # Non-depolarizing channels have no standard 2-qubit generalisation;
                # skip them here just as the default two-qubit path does below.
                continue
            channel = _two_qubit_depolarizing_channel(gate_prob)
            if channel is None:
                continue
            try:
                noise.add_all_qubit_channel(op, channel, num_controls=num_controls)
            except RuntimeError:
                # e.g. ``swap`` not supported as a noise hook on some targets
                continue
            overridden_two_qubit.add((op, num_controls))
            continue
        channel = _make_channel(config.noise_type, gate_prob)
        noise.add_all_qubit_channel(gate_name, channel)
        overridden_single.add(gate_name)

    # Default probability for all single-qubit gates not explicitly overridden.
    for gate_name in _DEFAULT_SINGLE_QUBIT_GATES:
        if gate_name not in overridden_single:
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
        two_q_channel = _two_qubit_depolarizing_channel(config.probability)
        if two_q_channel is not None:
            for op, num_controls in _DEFAULT_TWO_QUBIT_NOISE:
                if (op, num_controls) not in overridden_two_qubit:
                    noise.add_all_qubit_channel(
                        op, two_q_channel, num_controls=num_controls
                    )

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
