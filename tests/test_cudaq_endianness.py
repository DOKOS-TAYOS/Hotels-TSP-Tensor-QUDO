"""Integration test for CUDA-Q bitstring endianness.

Verifies that the bit-ordering convention assumed by our
``bitstring_to_qudit_sequence`` and ``bitstring_to_binary`` decoders
matches what ``cudaq.sample()`` actually returns.

Also includes GPU-level noise target selection tests (trajectory noise
when the ``nvidia`` target supports a ``noise_model`` kwarg).

Requires a working CUDA-Q installation with GPU access.
"""

from __future__ import annotations

import math

import pytest

cudaq = pytest.importorskip("cudaq")


def _has_gpu() -> bool:
    """Return True if CUDA-Q can detect an NVIDIA GPU."""
    try:
        return cudaq.num_available_gpus() >= 1 and cudaq.has_target("nvidia")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_gpu(),
    reason="CUDA-Q NVIDIA GPU target not available",
)


# ---------------------------------------------------------------------------
# QUBO endianness
# ---------------------------------------------------------------------------


class TestCudaqQuboEndianness:
    """Verify that QUBO bitstring decoding matches cudaq.sample() output."""

    def test_known_bitstring_from_x_gate(self) -> None:
        """Apply X to qubit 0 only → bitstring must have bit 0 set.

        If cudaq returns '100' (MSB-first) or '001' (LSB-first) for 3 qubits,
        our ``bitstring_to_binary`` must decode it so that x[0] == 1.
        """
        from solvers.cudaq_solver.cudaq_target import ensure_cudaq_target
        from solvers.cudaq_solver.qaoa_circuit_qubo import bitstring_to_binary

        ensure_cudaq_target()

        n_qubits = 3

        @cudaq.kernel
        def prep_kernel():
            q = cudaq.qvector(n_qubits)
            x(q[0])  # noqa: F821

        result = cudaq.sample(prep_kernel, shots_count=100)
        top_bitstring = result.most_probable()

        binary = bitstring_to_binary(top_bitstring)
        assert len(binary) == n_qubits
        # qubit 0 was flipped → x[0] must be 1
        assert binary[0] == 1, (
            f"Expected binary[0]==1 for X(q[0]). Got bitstring='{top_bitstring}', "
            f"decoded={binary.tolist()}. Endianness mismatch?"
        )
        # qubits 1,2 untouched → x[1], x[2] must be 0
        assert binary[1] == 0
        assert binary[2] == 0


# ---------------------------------------------------------------------------
# TQUDO endianness
# ---------------------------------------------------------------------------


class TestCudaqTqudoEndianness:
    """Verify that TQUDO qudit decoding matches cudaq.sample() output."""

    def test_known_qudit_from_x_gates(self) -> None:
        """Encode qudit value 1 in first qudit register (LSB encoding).

        With 2 qubits per qudit, value 1 = binary '01' (LSB first).
        After decoding, qudit 0 must be 1.
        """
        from solvers.cudaq_solver.cudaq_target import ensure_cudaq_target
        from solvers.cudaq_solver.qaoa_circuit_tqudo import bitstring_to_qudit_sequence

        ensure_cudaq_target()

        qubits_per_qudit = 2
        n_qudits = 2
        n_total = n_qudits * qubits_per_qudit  # 4 qubits

        @cudaq.kernel
        def prep_kernel():
            q = cudaq.qvector(n_total)
            # Encode qudit 0 = value 1 → set bit 0 of register 0
            x(q[0])  # noqa: F821
            # Encode qudit 1 = value 2 → set bit 1 of register 1
            x(q[qubits_per_qudit + 1])  # noqa: F821

        result = cudaq.sample(prep_kernel, shots_count=100)
        top_bitstring = result.most_probable()

        seq = bitstring_to_qudit_sequence(top_bitstring, n_qudits, qubits_per_qudit)

        assert seq[0] == 1, (
            f"Expected qudit[0]==1. Got bitstring='{top_bitstring}', "
            f"decoded={seq.tolist()}. Endianness mismatch?"
        )
        assert seq[1] == 2, (
            f"Expected qudit[1]==2. Got bitstring='{top_bitstring}', "
            f"decoded={seq.tolist()}. Endianness mismatch?"
        )


# ---------------------------------------------------------------------------
# Noise target selection (GPU integration)
# ---------------------------------------------------------------------------


class TestCudaqNoiseTargetSelection:
    """Verify GPU trajectory-based noise probe and noisy circuit execution."""

    def test_gpu_noise_probe_returns_bool(self) -> None:
        """``_gpu_supports_noise`` must return a boolean on a real GPU."""
        from solvers.cudaq_solver.cudaq_target import (
            _gpu_supports_noise,
            reset_target_state,
        )

        reset_target_state()
        result = _gpu_supports_noise()
        assert isinstance(result, bool)

    def test_ensure_target_with_noise_selects_valid_target(self) -> None:
        """With noise enabled, ``ensure_cudaq_target`` must pick a valid target."""
        from solvers.cudaq_solver.cudaq_target import (
            ensure_cudaq_target,
            get_current_target,
            reset_target_state,
        )
        from solvers.noise import NoiseConfig

        reset_target_state()
        noise = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)
        target = ensure_cudaq_target(noise)

        assert target in ("nvidia", "density-matrix-cpu")
        assert get_current_target() == target

    def test_noisy_bell_circuit_produces_samples(self) -> None:
        """A simple noisy 2-qubit circuit must sample without crashing."""
        from solvers.cudaq_solver.cudaq_target import (
            ensure_cudaq_target,
            reset_target_state,
        )
        from solvers.cudaq_solver.noise_model import get_noise_model
        from solvers.noise import NoiseConfig

        reset_target_state()
        noise = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.02)
        ensure_cudaq_target(noise)
        noise_model = get_noise_model(noise)

        @cudaq.kernel
        def bell():
            q = cudaq.qvector(2)
            h(q[0])  # noqa: F821
            x.ctrl(q[0], q[1])  # noqa: F821

        result = cudaq.sample(bell, shots_count=100, noise_model=noise_model)
        assert result.get_total_shots() == 100

    def test_idempotent_noisy_target(self) -> None:
        """Calling ``ensure_cudaq_target`` twice with noise should be idempotent."""
        from solvers.cudaq_solver.cudaq_target import (
            ensure_cudaq_target,
            reset_target_state,
        )
        from solvers.noise import NoiseConfig

        reset_target_state()
        noise = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)

        t1 = ensure_cudaq_target(noise)
        t2 = ensure_cudaq_target(noise)
        assert t1 == t2
