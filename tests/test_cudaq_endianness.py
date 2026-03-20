"""Integration test for CUDA-Q bitstring endianness.

Verifies that the bit-ordering convention assumed by our
``bitstring_to_qudit_sequence`` and ``bitstring_to_binary`` decoders
matches what ``cudaq.sample()`` actually returns.

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
