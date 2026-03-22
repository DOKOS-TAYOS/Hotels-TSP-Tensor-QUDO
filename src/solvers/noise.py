"""Shared noise configuration for quantum solver backends.

This module defines the :class:`NoiseConfig` dataclass that both the Cirq and
CUDA-Q solvers consume.  Each backend has its own ``noise_model.py`` that
translates a ``NoiseConfig`` into the library-specific noise objects.

Supported noise model types
----------------------------
- ``"depolarizing"`` — symmetric depolarizing channel (Pauli X/Y/Z with equal probability).
- ``"amplitude_damping"`` — energy relaxation (T₁-type decay).
- ``"phase_damping"`` — pure dephasing (T₂-type decay, no energy loss).
- ``"bit_flip"`` — random X errors.
- ``"phase_flip"`` — random Z errors.

When ``enabled`` is ``False`` (the default), the solvers behave exactly as in
the noiseless case — no simulator swap, no overhead.

Multi-qubit/qudit gate noise
-----------------------------
Non-depolarizing channels (amplitude_damping, phase_damping, bit_flip,
phase_flip) are inherently single-qubit/qudit phenomena with no standard
correlated multi-qubit generalisation.  The backends handle this as follows:

- **Cirq qubit** (d = 2): the single-qubit channel is applied
  **independently to each qubit** after every gate, including multi-qubit
  gates.
- **CUDA-Q**: non-depolarizing channels are applied **only to single-qubit
  gates**; two-qubit gates receive no noise for these channel types.  Only
  ``"depolarizing"`` is applied to two-qubit gates.
- **Cirq native qudit** (d > 2): two-qudit gates always receive a
  **correlated two-qudit depolarizing** channel (on the d²-dimensional
  joint space), regardless of the selected ``noise_type``.

If exact cross-backend comparability matters, use ``"depolarizing"``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

NoiseModelType = Literal[
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "bit_flip",
    "phase_flip",
]

VALID_NOISE_TYPES: frozenset[str] = frozenset(
    [
        "depolarizing",
        "amplitude_damping",
        "phase_damping",
        "bit_flip",
        "phase_flip",
    ]
)

# Warn the user when noise is enabled and qubit count exceeds this threshold
# because density-matrix simulators scale as O(4^n) in memory.
QUBIT_WARNING_THRESHOLD = 15


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Backend-agnostic noise simulation parameters.

    Attributes:
        enabled: Master switch — when ``False`` everything else is ignored.
        noise_type: Which noise channel to apply.
        probability: Error probability in [0, 1].  Meaning varies by channel:
            - depolarizing: probability of *any* Pauli error per gate.
            - amplitude_damping: decay probability γ.
            - phase_damping: dephasing probability γ.
            - bit_flip / phase_flip: single-Pauli error probability.
        gate_noise: Optional per-gate overrides mapping gate name →
            probability.  Gate names follow CUDA-Q conventions (lowercase):
            ``"x"``, ``"h"``, ``"rx"``, ``"rz"``, ``"cx"`` / ``"cnot"``, etc.
            When a gate is *not* listed here, ``probability`` is used as
            the fallback.
    """

    enabled: bool = False
    noise_type: NoiseModelType = "depolarizing"
    probability: float = 0.01
    gate_noise: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ``noise_type``, ``probability``, and per-gate probabilities.

        Raises:
            ValueError: If any field is out of range or ``noise_type`` is unknown.
        """
        if self.noise_type not in VALID_NOISE_TYPES:
            raise ValueError(
                f"noise_type must be one of {sorted(VALID_NOISE_TYPES)}, "
                f"got: {self.noise_type!r}"
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"probability must be in [0, 1], got {self.probability}"
            )
        for gate, prob in self.gate_noise.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"gate_noise[{gate!r}] probability must be in [0, 1], got {prob}"
                )

    def warn_if_large_system(
        self,
        n_qubits: int,
        *,
        gpu_trajectory: bool = False,
        qudit_dimension: int = 2,
    ) -> None:
        """Log a performance warning when the system exceeds the safe threshold.

        Args:
            n_qubits: Total number of quantum systems (qubits *or* qudits).
            gpu_trajectory: If ``True``, the backend uses GPU trajectory-based
                simulation (O(2ⁿ)) instead of density-matrix (O(4ⁿ)).
            qudit_dimension: Dimension of each quantum system.  For qudits
                (d > 2) the effective Hilbert-space size is d^n, so the
                equivalent qubit count is ``n * log₂(d)``.
        """
        if not self.enabled:
            return
        effective_qubits = n_qubits * math.log2(max(qudit_dimension, 2))
        if effective_qubits <= QUBIT_WARNING_THRESHOLD:
            return
        if qudit_dimension > 2:
            logger.warning(
                "Noise simulation enabled with %d qudits (dimension=%d, "
                "~%.0f equivalent qubits). Density-matrix memory scales "
                "as O(d^{2n}) — expect significantly higher resource usage.",
                n_qubits,
                qudit_dimension,
                effective_qubits,
            )
        elif gpu_trajectory:
            logger.warning(
                "Noise simulation enabled with %d qubits on the GPU "
                "trajectory backend. Memory scales as O(2^n) — still "
                "significant for large circuits.",
                n_qubits,
            )
        else:
            logger.warning(
                "Noise simulation enabled with %d qubits on the "
                "density-matrix backend. Memory scales as O(4^n) — "
                "expect significantly higher resource usage.",
                n_qubits,
            )
