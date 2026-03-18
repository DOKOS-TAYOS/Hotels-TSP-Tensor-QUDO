"""QUBO to Ising conversion helpers shared by quantum backends."""

from __future__ import annotations

import numpy as np


def qubo_to_ising(qubo_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a symmetric QUBO matrix to Ising form with offset.

    This helper assumes the QUBO objective is evaluated as ``x.T @ Q @ x`` for a
    symmetric matrix ``Q``. With the substitution ``x_i = (1 - s_i) / 2`` and
    ``s_i in {-1, +1}``, the energy relation is:

    ``E_QUBO(x) = E_Ising(s) + offset``

    where:
    - ``h[i] = -0.5 * sum_j Q[i, j]``
    - ``J[i, j] = Q[i, j] / 2`` for ``i < j``
    - ``offset = trace(Q) / 2 + sum_{i < j} Q[i, j] / 2``

    Args:
        qubo_matrix: Symmetric matrix of shape ``(n, n)``.

    Returns:
        Tuple ``(h, J, offset)`` where ``J`` is upper triangular.

    Raises:
        ValueError: If ``qubo_matrix`` is not square or symmetric.
    """
    n = qubo_matrix.shape[0]
    if qubo_matrix.shape[1] != n:
        raise ValueError("qubo_matrix must be square")
    if not np.allclose(qubo_matrix, qubo_matrix.T):
        raise ValueError("qubo_matrix must be symmetric")

    h = -0.5 * np.sum(qubo_matrix, axis=1)
    j_full = np.triu(qubo_matrix, k=1) / 2.0
    offset = 0.5 * np.trace(qubo_matrix) + 0.5 * np.sum(np.triu(qubo_matrix, k=1))

    return h, j_full, float(offset)
