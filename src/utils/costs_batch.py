"""Vectorised QUBO / TQUDO cost evaluation (same algebra as :mod:`utils.costs`)."""

from __future__ import annotations

import numpy as np


def unpack_qubo_bitmatrix(i_vals: np.ndarray, n_vars: int) -> np.ndarray:
    """Decode integer indices to QUBO bit rows; shape ``(len(i_vals), n_vars)`` float {0,1}."""
    i_vals = np.asarray(i_vals, dtype=np.int64)
    b_idx = np.arange(n_vars, dtype=np.int64)
    return ((i_vals[:, None] >> b_idx) & 1).astype(np.float64)


def batch_qubo_costs(
    qubo_matrix: np.ndarray, energy_scale: float, x_bits: np.ndarray
) -> np.ndarray:
    """Vectorized ``x @ Q @ x`` per row; *x_bits* shape ``(B, n_vars)``."""
    q = np.asarray(qubo_matrix, dtype=np.float64)
    return np.sum((x_bits @ q) * x_bits, axis=1) * energy_scale


def bitstrings_to_binary_matrix(bitstrings: list[str]) -> np.ndarray:
    """Decode ``'0'``/``'1'`` strings to rows of floats; shape ``(len(bitstrings), n_bits)``.

    All strings must have equal length (CUDA-Q / emulation histogram keys).
    """
    if not bitstrings:
        return np.zeros((0, 0), dtype=np.float64)
    n_bits = len(bitstrings[0])
    out = np.empty((len(bitstrings), n_bits), dtype=np.float64)
    for i, s in enumerate(bitstrings):
        out[i] = np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.float64) - 48.0
    return out


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
    costs += np.sum(
        ettprimeab[t_left, t_right, sequences[:, t_left], sequences[:, t_right]],
        axis=1,
    )
    return costs * energy_scale


def bit_rows_to_qudit_sequences(
    bits: np.ndarray,
    n_qudits: int,
    qubits_per_qudit: int,
) -> np.ndarray:
    """Decode measured bit rows (``0``/``1``) to qudit city indices; shape ``(B, n_qudits)``."""
    bits = np.asarray(bits, dtype=np.float64)
    b, n_bits = bits.shape
    expected = n_qudits * qubits_per_qudit
    if n_bits != expected:
        raise ValueError(f"bit row length {n_bits} != n_qudits * qubits_per_qudit = {expected}")
    flat = bits.reshape(b, n_qudits, qubits_per_qudit)
    weights = (1 << np.arange(qubits_per_qudit, dtype=np.int64)).astype(np.float64)
    weights = weights[None, None, :]
    return np.sum(flat * weights, axis=2).astype(np.int64)


def bitstring_to_qudit_sequence(
    bitstring: str,
    n_qudits: int,
    qubits_per_qudit: int,
) -> np.ndarray:
    """Decode a contiguous ``'0'``/``'1'`` measurement string to qudit indices (little-endian per block).

    Same convention as :func:`bit_rows_to_qudit_sequences` for a single row.
    """
    mat = bitstrings_to_binary_matrix([bitstring])
    return bit_rows_to_qudit_sequences(mat, n_qudits, qubits_per_qudit)[0]


def qudit_sequence_to_bitstring(
    sequence: np.ndarray | list[int] | tuple[int, ...],
    qubits_per_qudit: int,
) -> str:
    """Encode qudit indices as a contiguous ``'0'``/``'1'`` string (inverse of :func:`bitstring_to_qudit_sequence`).

    Within each qudit block, bits are little-endian (column ``k`` has weight ``2**k``), matching
    :func:`bit_rows_to_qudit_sequences`.
    """
    seq = np.asarray(sequence, dtype=np.int64).ravel()
    parts: list[str] = []
    for v in seq:
        vi = int(v)
        for k in range(int(qubits_per_qudit)):
            parts.append(str((vi >> k) & 1))
    return "".join(parts)
