"""Vectorised QUBO / TQUDO cost evaluation (same algebra as :mod:`utils.costs`)."""

from __future__ import annotations

import numpy as np


def unpack_qubo_bitmatrix(i_vals: np.ndarray, n_vars: int) -> np.ndarray:
    """Decode integer indices to QUBO bit rows; shape ``(len(i_vals), n_vars)`` float {0,1}."""
    i_vals = np.asarray(i_vals, dtype=np.int64)
    b_idx = np.arange(n_vars, dtype=np.int64)
    return ((i_vals[:, None] >> b_idx) & 1).astype(np.float64)


def batch_qubo_costs(qubo_matrix: np.ndarray, energy_scale: float, x_bits: np.ndarray) -> np.ndarray:
    """Vectorized ``x @ Q @ x`` per row; *x_bits* shape ``(B, n_vars)``."""
    q = np.asarray(qubo_matrix, dtype=np.float64)
    return np.sum((x_bits @ q) * x_bits, axis=1) * energy_scale


def unpack_tqudo_sequences(i_vals: np.ndarray, n_available: int) -> np.ndarray:
    """Mixed-radix digits; shape ``(len(i_vals), n_available)`` int64."""
    rem = np.asarray(i_vals, dtype=np.int64).copy()
    n = n_available
    out = np.empty((len(rem), n), dtype=np.int64)
    for t in range(n):
        out[:, t] = rem % n
        rem //= n
    return out


def batch_tqudo_costs(
    etab: np.ndarray,
    ettprimeab: np.ndarray,
    sequences: np.ndarray,
    energy_scale: float,
) -> np.ndarray:
    """Batched TQUDO objective (same algebra as :func:`~utils.costs.calculate_tqudo_cost`)."""
    batch_len, n = sequences.shape
    costs = np.zeros(batch_len, dtype=np.float64)
    for t in range(n - 1):
        costs += etab[t, sequences[:, t], sequences[:, t + 1]]
    t_left, t_right = np.triu_indices(n, k=1)
    for k in range(t_left.size):
        tl = int(t_left[k])
        tr = int(t_right[k])
        costs += ettprimeab[tl, tr, sequences[:, tl], sequences[:, tr]]
    return costs * energy_scale
