"""Custom d-dimensional noise channels and noise model for native Cirq qudits.

Cirq's built-in noise channels (``depolarize``, ``amplitude_damp``, etc.) are
qubit-only (2×2 Kraus operators).  This module provides d-dimensional
generalizations that implement the ``_kraus_`` protocol, which
``cirq.DensityMatrixSimulator`` handles natively for any qudit dimension.

Channel generalizations
-----------------------
- **Depolarizing**: ρ → (1−p)ρ + (p/d)I — uniform mixture toward the
  maximally mixed state.
- **Amplitude damping**: cascade decay — every excited level |k⟩ (k > 0)
  decays to |0⟩ with probability γ.
- **Phase damping**: off-diagonal dephasing — coherences with |0⟩ decay by
  √(1−γ), coherences between excited states decay by (1−γ), populations
  unchanged.  For d = 2 only |0⟩-coherences exist, so both descriptions
  coincide.
- **Bit flip**: cyclic shift X_d applied with probability p.
- **Phase flip**: clock matrix Z_d = diag(1, ω, ω², …) applied with
  probability p, where ω = exp(2πi/d).

Two-qudit correlated noise
--------------------------
For two-qudit gates (e.g. ``QuditDiagonalCostGate``), a **correlated**
depolarizing channel is applied on the full d²-dimensional subspace:
ρ → (1−p)ρ + (p/d²)I_{d²}.
"""

from __future__ import annotations

import numpy as np

import cirq

from solvers.noise import NoiseConfig


# ---------------------------------------------------------------------------
# Gate class name → gate_noise key mapping.
# Using class names (strings) avoids a circular import with
# qaoa_circuit_tqudo.py.
# ---------------------------------------------------------------------------
_GATE_NAME_TO_KEY: dict[str, str] = {
    "QuditHadamardGate": "qudit_hadamard",
    "QuditDiagonalCostGate": "qudit_cost",
    "QuditRingMixerGate": "qudit_mixer",
}


# ═══════════════════════════════════════════════════════════════════════════
# Single-qudit noise channels
# ═══════════════════════════════════════════════════════════════════════════


class QuditDepolarizingChannel(cirq.Gate):
    r"""d-dimensional depolarizing: ρ → (1−p)ρ + (p/d)I_d.

    Kraus operators:
      - K_0 = √(1−p) · I_d
      - K_{ij} = √(p/d) · |i⟩⟨j|   for i, j = 0, …, d−1

    Total: 1 + d² operators.  For d = 2 this reduces to the standard qubit
    depolarizing channel with the same parameterisation.
    """

    def __init__(self, dimension: int, p: float) -> None:
        super().__init__()
        self._d = dimension
        self._p = p

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._d,)

    def _kraus_(self) -> list[np.ndarray]:
        d, p = self._d, self._p
        ops: list[np.ndarray] = [np.sqrt(1.0 - p) * np.eye(d, dtype=complex)]
        coeff = np.sqrt(p / d)
        for i in range(d):
            for j in range(d):
                k = np.zeros((d, d), dtype=complex)
                k[i, j] = coeff
                ops.append(k)
        return ops

    def __repr__(self) -> str:
        return f"QuditDepolarizingChannel(d={self._d}, p={self._p})"


class QuditAmplitudeDampingChannel(cirq.Gate):
    r"""d-dimensional amplitude damping: |k⟩ → |0⟩ with prob γ for k > 0.

    Kraus operators:
      - K_0 = |0⟩⟨0| + √(1−γ) Σ_{k≥1} |k⟩⟨k|
      - K_k = √γ · |0⟩⟨k|   for k = 1, …, d−1

    For d = 2 this is the standard amplitude-damping channel.
    """

    def __init__(self, dimension: int, gamma: float) -> None:
        super().__init__()
        self._d = dimension
        self._gamma = gamma

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._d,)

    def _kraus_(self) -> list[np.ndarray]:
        d, g = self._d, self._gamma
        k0 = np.zeros((d, d), dtype=complex)
        k0[0, 0] = 1.0
        for k in range(1, d):
            k0[k, k] = np.sqrt(1.0 - g)
        ops: list[np.ndarray] = [k0]
        for k in range(1, d):
            kk = np.zeros((d, d), dtype=complex)
            kk[0, k] = np.sqrt(g)
            ops.append(kk)
        return ops

    def __repr__(self) -> str:
        return f"QuditAmplitudeDampingChannel(d={self._d}, γ={self._gamma})"


class QuditPhaseDampingChannel(cirq.Gate):
    r"""d-dimensional phase damping (generalised pure dephasing).

    Kraus operators:
      - K_0 = |0⟩⟨0| + √(1−γ) Σ_{k≥1} |k⟩⟨k|
      - K_k = √γ · |k⟩⟨k|   for k = 1, …, d−1

    Populations are unchanged.  All off-diagonal elements are suppressed:
    coherences involving |0⟩ decay by √(1−γ), while coherences between
    excited states (m, n ≥ 1, m ≠ n) decay by (1−γ).  For d = 2 only
    |0⟩-coherences exist, so this reduces to the standard qubit
    phase-damping channel.
    """

    def __init__(self, dimension: int, gamma: float) -> None:
        super().__init__()
        self._d = dimension
        self._gamma = gamma

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._d,)

    def _kraus_(self) -> list[np.ndarray]:
        d, g = self._d, self._gamma
        k0 = np.zeros((d, d), dtype=complex)
        k0[0, 0] = 1.0
        for k in range(1, d):
            k0[k, k] = np.sqrt(1.0 - g)
        ops: list[np.ndarray] = [k0]
        for k in range(1, d):
            kk = np.zeros((d, d), dtype=complex)
            kk[k, k] = np.sqrt(g)
            ops.append(kk)
        return ops

    def __repr__(self) -> str:
        return f"QuditPhaseDampingChannel(d={self._d}, γ={self._gamma})"


class QuditBitFlipChannel(cirq.Gate):
    r"""d-dimensional bit flip: cyclic shift X_d applied with probability p.

    X_d|k⟩ = |k+1 mod d⟩ — the generalised Pauli-X.

    Kraus operators:
      - K_0 = √(1−p) · I_d
      - K_1 = √p · X_d
    """

    def __init__(self, dimension: int, p: float) -> None:
        super().__init__()
        self._d = dimension
        self._p = p

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._d,)

    def _kraus_(self) -> list[np.ndarray]:
        d, p = self._d, self._p
        x_d = np.zeros((d, d), dtype=complex)
        for k in range(d):
            x_d[(k + 1) % d, k] = 1.0
        return [
            np.sqrt(1.0 - p) * np.eye(d, dtype=complex),
            np.sqrt(p) * x_d,
        ]

    def __repr__(self) -> str:
        return f"QuditBitFlipChannel(d={self._d}, p={self._p})"


class QuditPhaseFlipChannel(cirq.Gate):
    r"""d-dimensional phase flip: clock Z_d applied with probability p.

    Z_d = diag(1, ω, ω², …, ω^{d−1})  where ω = exp(2πi/d).

    Kraus operators:
      - K_0 = √(1−p) · I_d
      - K_1 = √p · Z_d

    For d = 2, ω = −1 and Z_d = diag(1, −1) = Pauli-Z.
    """

    def __init__(self, dimension: int, p: float) -> None:
        super().__init__()
        self._d = dimension
        self._p = p

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._d,)

    def _kraus_(self) -> list[np.ndarray]:
        d, p = self._d, self._p
        omega = np.exp(2j * np.pi / d)
        z_d = np.diag(np.array([omega**k for k in range(d)], dtype=complex))
        return [
            np.sqrt(1.0 - p) * np.eye(d, dtype=complex),
            np.sqrt(p) * z_d,
        ]

    def __repr__(self) -> str:
        return f"QuditPhaseFlipChannel(d={self._d}, p={self._p})"


# ═══════════════════════════════════════════════════════════════════════════
# Two-qudit correlated noise channel
# ═══════════════════════════════════════════════════════════════════════════


class TwoQuditDepolarizingChannel(cirq.Gate):
    r"""Correlated two-qudit depolarizing: ρ → (1−p)ρ + (p/d²)I_{d²}.

    Acts on two qudits of dimension *d* (joint Hilbert-space dimension d²).

    Kraus operators:
      - K_0 = √(1−p) · I_{d²}
      - K_{mn} = √(p/d²) · |m⟩⟨n|   for m, n = 0, …, d²−1

    This is the standard depolarizing channel on the *joint* d²-dimensional
    space.  It introduces **correlated** noise: neither qudit is treated
    independently.
    """

    def __init__(self, dimension: int, p: float) -> None:
        super().__init__()
        self._d = dimension
        self._p = p

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._d, self._d)

    def _kraus_(self) -> list[np.ndarray]:
        d2 = self._d**2
        p = self._p
        ops: list[np.ndarray] = [np.sqrt(1.0 - p) * np.eye(d2, dtype=complex)]
        coeff = np.sqrt(p / d2)
        for m in range(d2):
            for n in range(d2):
                k = np.zeros((d2, d2), dtype=complex)
                k[m, n] = coeff
                ops.append(k)
        return ops

    def __repr__(self) -> str:
        return f"TwoQuditDepolarizingChannel(d={self._d}, p={self._p})"


# ═══════════════════════════════════════════════════════════════════════════
# Channel factories
# ═══════════════════════════════════════════════════════════════════════════

_QUDIT_CHANNEL_FACTORIES: dict[str, type] = {
    "depolarizing": QuditDepolarizingChannel,
    "amplitude_damping": QuditAmplitudeDampingChannel,
    "phase_damping": QuditPhaseDampingChannel,
    "bit_flip": QuditBitFlipChannel,
    "phase_flip": QuditPhaseFlipChannel,
}


def _make_single_qudit_channel(
    noise_type: str,
    dimension: int,
    probability: float,
) -> cirq.Gate:
    """Create a single-qudit noise channel gate.

    Raises:
        ValueError: If *noise_type* is not recognised.
    """
    factory = _QUDIT_CHANNEL_FACTORIES.get(noise_type)
    if factory is None:
        raise ValueError(f"Unknown qudit noise type: {noise_type!r}")
    return factory(dimension, probability)


def _make_two_qudit_channel(
    noise_type: str,  # noqa: ARG001 – kept for API symmetry
    dimension: int,
    probability: float,
) -> cirq.Gate:
    """Create a correlated two-qudit noise channel.

    Regardless of *noise_type*, a **two-qudit depolarizing** channel is used.
    Generalising amplitude-damping or phase-damping to the correlated
    two-qudit setting is non-trivial; depolarizing is the standard choice for
    multi-qudit correlated noise.
    """
    return TwoQuditDepolarizingChannel(dimension, probability)


# ═══════════════════════════════════════════════════════════════════════════
# Custom NoiseModel for qudit circuits
# ═══════════════════════════════════════════════════════════════════════════


class ConstantQuditNoiseModel(cirq.NoiseModel):
    """Noise model for native qudit circuits.

    Applies dimension-matched noise channels after every gate operation:

    - **Single-qudit gates** → single-qudit channel (type from
      ``NoiseConfig.noise_type``).
    - **Two-qudit gates** → correlated two-qudit depolarizing channel.
    - **3+-qudit gates** (unlikely) → independent single-qudit channels on
      each qudit.

    Per-gate probability overrides are supported via qudit-specific keys in
    ``NoiseConfig.gate_noise``:

    - ``"qudit_hadamard"`` → ``QuditHadamardGate``
    - ``"qudit_mixer"``    → ``QuditRingMixerGate``
    - ``"qudit_cost"``     → ``QuditDiagonalCostGate``
    """

    def __init__(self, config: NoiseConfig, dimension: int) -> None:
        self._config = config
        self._dimension = dimension

    # ------------------------------------------------------------------

    def _get_probability(self, gate: cirq.Gate | None) -> float:
        """Look up the error probability for *gate*, checking overrides."""
        if gate is not None:
            key = _GATE_NAME_TO_KEY.get(type(gate).__name__)
            if key is not None and key in self._config.gate_noise:
                return self._config.gate_noise[key]
        return self._config.probability

    # ------------------------------------------------------------------

    def noisy_operation(self, op: cirq.Operation) -> cirq.OP_TREE:
        """Append noise after every non-measurement gate operation."""
        if isinstance(op.gate, cirq.MeasurementGate) or op.gate is None:
            return op

        prob = self._get_probability(op.gate)
        qids = op.qubits
        n = len(qids)

        if n == 1:
            ch = _make_single_qudit_channel(
                self._config.noise_type, self._dimension, prob,
            )
            return [op, ch.on(qids[0])]

        if n == 2:
            ch = _make_two_qudit_channel(
                self._config.noise_type, self._dimension, prob,
            )
            return [op, ch.on(qids[0], qids[1])]

        # Fallback: independent single-qudit noise on each qudit.
        noise_ops = [
            _make_single_qudit_channel(
                self._config.noise_type, self._dimension, prob,
            ).on(q)
            for q in qids
        ]
        return [op, *noise_ops]


# ═══════════════════════════════════════════════════════════════════════════
# Public builder
# ═══════════════════════════════════════════════════════════════════════════


def build_qudit_noise_model(
    config: NoiseConfig,
    dimension: int,
) -> ConstantQuditNoiseModel:
    """Build a :class:`ConstantQuditNoiseModel` from a :class:`NoiseConfig`.

    This is the entry-point called by
    :func:`solvers.cirq_solver.noise_model.build_noise_model` when the qudit
    dimension is > 2.
    """
    return ConstantQuditNoiseModel(config, dimension)
