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
* For **qubit** circuits (``qudit_dimension=2``), a custom
  :class:`ConstantQubitNoiseModelWithOverrides` applies per-gate probability
  overrides from ``NoiseConfig.gate_noise``.
* For **native qudit** circuits (``qudit_dimension > 2``), custom
  d-dimensional channels from :mod:`solvers.cirq_solver.qudit_noise_channels`
  are used via :class:`ConstantQuditNoiseModel`.  Two-qudit gates receive
  correlated depolarizing noise on the full d²-dimensional subspace.
"""

from __future__ import annotations

from collections.abc import Callable

import cirq

from solvers.noise import NoiseConfig


_CHANNEL_FACTORIES: dict[str, Callable[[float], cirq.Gate]] = {
    "depolarizing": lambda p: cirq.depolarize(p=p),
    "amplitude_damping": lambda p: cirq.amplitude_damp(gamma=p),
    "phase_damping": lambda p: cirq.phase_damp(gamma=p),
    "bit_flip": lambda p: cirq.bit_flip(p=p),
    "phase_flip": lambda p: cirq.phase_flip(p=p),
}

# Mapping from Cirq gate class names to the lowercase gate keys used in
# NoiseConfig.gate_noise.  Extends naturally as new gates are used.
_CIRQ_GATE_NAME_TO_KEY: dict[str, str] = {
    "HPowGate": "h",
    "XPowGate": "x",
    "YPowGate": "y",
    "ZPowGate": "z",
    "Rx": "rx",
    "Ry": "ry",
    "Rz": "rz",
    "CXPowGate": "cx",
    "CZPowGate": "cz",
    "CNotPowGate": "cx",
    "SwapPowGate": "swap",
    "MeasurementGate": "",  # never noised
}


class ConstantQubitNoiseModelWithOverrides(cirq.NoiseModel):
    """Qubit noise model that supports per-gate probability overrides.

    Unlike ``cirq.ConstantQubitNoiseModel`` which applies a single noise
    channel to every gate uniformly, this model checks
    ``NoiseConfig.gate_noise`` for per-gate probability overrides before
    falling back to ``NoiseConfig.probability``.

    For multi-qubit gates the single-qubit noise channel is applied
    **independently** to each qubit involved in the operation.

    Args:
        config: Global noise type, default probability, and per-gate overrides.

    """

    def __init__(self, config: NoiseConfig) -> None:
        self._config = config

    def _get_probability(self, gate: cirq.Gate | None) -> float:
        """Return the error probability for *gate*, using overrides when set.

        Args:
            gate: Gate about to run, or None.

        Returns:
            Probability in ``[0, 1]`` from ``gate_noise`` or ``config.probability``.

        """
        if gate is not None:
            key = _CIRQ_GATE_NAME_TO_KEY.get(type(gate).__name__)
            if key is not None and key in self._config.gate_noise:
                return self._config.gate_noise[key]
        return self._config.probability

    def noisy_operation(self, op: cirq.Operation) -> cirq.OP_TREE:
        """Return *op* followed by single-qubit noise on each operand qubit.

        Args:
            op: Circuit operation (skipped for measurements).

        Returns:
            Operation tree: gate plus noise channels.

        Raises:
            ValueError: If ``noise_type`` is not supported for qubits.

        """
        if isinstance(op.gate, cirq.MeasurementGate) or op.gate is None:
            return op

        prob = self._get_probability(op.gate)
        factory = _CHANNEL_FACTORIES.get(self._config.noise_type)
        if factory is None:
            raise ValueError(
                f"Unknown noise type for Cirq: {self._config.noise_type!r}"
            )

        noise_gate = factory(prob)
        noise_ops = [noise_gate.on(q) for q in op.qubits]
        return [op, *noise_ops]


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

    return ConstantQubitNoiseModelWithOverrides(config)


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
