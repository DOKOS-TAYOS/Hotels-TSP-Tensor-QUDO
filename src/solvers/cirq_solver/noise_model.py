"""Build Cirq noise models and select the appropriate simulator.

Translates a :class:`NoiseConfig` into Cirq-native objects so that the QAOA
circuit modules only need to call :func:`get_simulator` instead of managing
noise details themselves.

Design notes
------------
* When noise is **disabled**, :func:`get_simulator` returns a plain
  ``cirq.Simulator`` (state-vector) — identical to the previous behaviour.
* When noise is **enabled**, it returns a ``cirq.DensityMatrixSimulator`` and
  the caller must wrap the circuit with ``circuit.with_noise(noise_model)``
  before running it.
* For **qubit** circuits (``qudit_dimension=2``), the standard Cirq built-in
  channels (``cirq.depolarize``, ``cirq.amplitude_damp``, etc.) are used via
  ``cirq.ConstantQubitNoiseModel``.
* For **native qudit** circuits (``qudit_dimension > 2``), custom
  d-dimensional channels from :mod:`solvers.cirq_solver.qudit_noise_channels`
  are used via :class:`ConstantQuditNoiseModel`.  Two-qudit gates receive
  correlated depolarizing noise on the full d²-dimensional subspace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cirq

from solvers.noise import NoiseConfig

if TYPE_CHECKING:
    pass


# Mapping from NoiseConfig.noise_type to a factory that returns a cirq gate
# (a single-qubit channel).  Each factory takes (probability: float) -> cirq.Gate.
_CHANNEL_FACTORIES: dict[str, type] = {
    "depolarizing": lambda p: cirq.depolarize(p=p),
    "amplitude_damping": lambda p: cirq.amplitude_damp(gamma=p),
    "phase_damping": lambda p: cirq.phase_damp(gamma=p),
    "bit_flip": lambda p: cirq.bit_flip(p=p),
    "phase_flip": lambda p: cirq.phase_flip(p=p),
}


def build_noise_model(
    config: NoiseConfig,
    *,
    qudit_dimension: int = 2,
) -> cirq.NOISE_MODEL_LIKE:
    """Construct a Cirq ``NoiseModel`` from a :class:`NoiseConfig`.

    Args:
        config: The noise parameters.
        qudit_dimension: Dimension of the qudits in the circuit.  For
            ``qudit_dimension > 2`` a custom :class:`ConstantQuditNoiseModel`
            with d-dimensional Kraus operators is used.

    Returns:
        A ``cirq.NoiseModel`` ready to be passed to
        ``circuit.with_noise(…)``.

    Raises:
        ValueError: If ``config.noise_type`` is unknown.
    """
    if qudit_dimension > 2:
        from solvers.cirq_solver.qudit_noise_channels import build_qudit_noise_model

        return build_qudit_noise_model(config, qudit_dimension)

    factory = _CHANNEL_FACTORIES.get(config.noise_type)
    if factory is None:
        raise ValueError(f"Unknown noise type for Cirq: {config.noise_type!r}")

    noise_gate = factory(config.probability)
    return cirq.ConstantQubitNoiseModel(qubit_noise_gate=noise_gate)


def get_simulator(
    config: NoiseConfig | None,
    *,
    qudit_dimension: int = 2,
    seed: int | None = None,
) -> tuple[cirq.SimulatesSamples, cirq.NOISE_MODEL_LIKE | None]:
    """Return the right (simulator, noise_model) pair.

    When noise is disabled (or *config* is ``None``) the function returns a
    fast state-vector ``cirq.Simulator`` and ``noise_model=None``.

    When noise is enabled it returns ``cirq.DensityMatrixSimulator`` together
    with the constructed ``NoiseModel``.  The caller is responsible for
    attaching the noise model to the circuit via
    ``circuit.with_noise(noise_model)``.

    Args:
        config: Noise parameters (``None`` ≡ disabled).
        qudit_dimension: Passed through to :func:`build_noise_model`.
        seed: Random seed forwarded to the simulator.

    Returns:
        ``(simulator, noise_model)`` — *noise_model* is ``None`` when noise
        is off.
    """
    if config is None or not config.enabled:
        return cirq.Simulator(seed=seed), None

    noise_model = build_noise_model(config, qudit_dimension=qudit_dimension)
    simulator = cirq.DensityMatrixSimulator(seed=seed)
    return simulator, noise_model
