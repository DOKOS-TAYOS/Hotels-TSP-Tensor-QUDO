"""Parity tests: batched cost helpers vs scalar QUBO/TQUDO objectives."""

from __future__ import annotations

import numpy as np
import pytest

from instance_gen_process.models import ProblemTQUDO
from utils.costs import calculate_tqudo_cost
from utils.costs_batch import (
    batch_qubo_costs,
    batch_tqudo_costs,
    bit_rows_to_qudit_sequences,
    bitstring_to_qudit_sequence,
    bitstrings_to_binary_matrix,
    qudit_sequence_to_bitstring,
)


class TestBatchQuboParity:
    """batch_qubo_costs matches x @ Q @ x per row."""

    def test_random_bit_rows(self) -> None:
        rng = np.random.default_rng(42)
        n_vars = 12
        q = rng.standard_normal((n_vars, n_vars))
        q = (q + q.T) / 2.0
        x = rng.integers(0, 2, size=(50, n_vars)).astype(np.float64)
        batched = batch_qubo_costs(q, 1.0, x)
        for i in range(x.shape[0]):
            expected = float(x[i] @ q @ x[i])
            assert batched[i] == pytest.approx(expected)


class TestBatchTqudoParity:
    """batch_tqudo_costs matches calculate_tqudo_cost per row."""

    def test_random_sequences(self) -> None:
        rng = np.random.default_rng(7)
        n = 6
        d = 5
        Etab = rng.standard_normal((n, d, d))
        Ett = rng.standard_normal((n, n, d, d))
        es = 1.15
        problem = ProblemTQUDO(Etab=Etab, Ettprimeab=Ett, energy_scale=es)
        seqs = rng.integers(0, d, size=(30, n))
        batched = batch_tqudo_costs(Etab, Ett, seqs, es)
        for i in range(seqs.shape[0]):
            expected = calculate_tqudo_cost(problem, seqs[i])
            assert batched[i] == pytest.approx(expected)


class TestQuditSequenceBitstringRoundTrip:
    """Encode then decode recovers qudit indices."""

    def test_round_trip_various_shapes(self) -> None:
        rng = np.random.default_rng(1)
        for n_qudits, qubits_per_qudit in [(3, 2), (4, 2), (5, 3)]:
            max_val = (1 << qubits_per_qudit) - 1
            for _ in range(30):
                seq = rng.integers(0, max_val + 1, size=n_qudits, dtype=np.int64)
                bitstring = qudit_sequence_to_bitstring(seq, qubits_per_qudit)
                decoded = bitstring_to_qudit_sequence(bitstring, n_qudits, qubits_per_qudit)
                assert np.array_equal(decoded, seq)

    def test_known_n5_example(self) -> None:
        seq = np.array([0, 2, 1, 3], dtype=np.int64)
        assert qudit_sequence_to_bitstring(seq, 2) == "00011011"
        assert np.array_equal(
            bitstring_to_qudit_sequence("00011011", 4, 2),
            seq,
        )


class TestBitstringDecodeParity:
    """Batch bit decoding matches little-endian qudit blocks."""

    def test_bitstrings_to_matrix_and_qudits(self) -> None:
        rng = np.random.default_rng(0)
        for nq, qb in [(3, 2), (5, 3)]:
            for _ in range(20):
                s = "".join(str(rng.integers(0, 2)) for _ in range(nq * qb))
                mat = bitstrings_to_binary_matrix([s])
                seq = bit_rows_to_qudit_sequences(mat, nq, qb)[0]
                ref = bitstring_to_qudit_sequence(s, nq, qb)
                assert np.array_equal(seq, ref)
