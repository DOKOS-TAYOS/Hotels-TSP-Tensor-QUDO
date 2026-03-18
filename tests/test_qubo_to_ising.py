"""Tests for shared QUBO-to-Ising conversion helpers."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from math_utils import qubo_to_ising


def _ising_energy(
    h: np.ndarray,
    j_matrix: np.ndarray,
    offset: float,
    binary_solution: np.ndarray,
) -> float:
    """Evaluate the Ising energy plus offset for a binary assignment."""
    spins = 1 - 2 * binary_solution
    pair_energy = 0.0
    n = len(spins)
    for i in range(n):
        for j in range(i + 1, n):
            pair_energy += float(j_matrix[i, j] * spins[i] * spins[j])
    return float(np.dot(h, spins) + pair_energy + offset)


@pytest.mark.parametrize(
    "qubo_matrix",
    [
        np.array([[2.0, 1.0], [1.0, 4.0]], dtype=float),
        np.array(
            [
                [3.0, -1.0, 0.5],
                [-1.0, 2.0, 1.5],
                [0.5, 1.5, 5.0],
            ],
            dtype=float,
        ),
    ],
)
def test_qubo_to_ising_preserves_energy_for_all_binary_assignments(
    qubo_matrix: np.ndarray,
) -> None:
    """E_QUBO(x) must equal E_Ising(s) + offset for every binary assignment."""
    h, j_matrix, offset = qubo_to_ising(qubo_matrix)
    n = qubo_matrix.shape[0]

    for bits in product([0, 1], repeat=n):
        binary_solution = np.array(bits, dtype=np.int64)
        qubo_energy = float(binary_solution @ qubo_matrix @ binary_solution)
        ising_energy = _ising_energy(h, j_matrix, offset, binary_solution)
        assert ising_energy == pytest.approx(qubo_energy)


def test_qubo_to_ising_rejects_non_symmetric_matrix() -> None:
    """The helper must reject non-symmetric QUBO matrices."""
    with pytest.raises(ValueError, match="symmetric"):
        qubo_to_ising(np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float))
