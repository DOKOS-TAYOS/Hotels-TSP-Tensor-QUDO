"""Shared helpers used across QAOA circuit implementations."""

from __future__ import annotations

import numpy as np


def tqa_init_params(depth: int, delta_t: float) -> np.ndarray:
    """Return TQA (Trotterized Quantum Annealing) initial parameters.

    gamma_i = (i / p) * delta_t,  beta_i = (1 - i / p) * delta_t
    for i = 1, ..., p.

    Returns:
        1-D array of shape (2 * depth,) with [gamma_1..gamma_p, beta_1..beta_p].
    """
    indices = np.arange(1, depth + 1)
    gamma_init = (indices / depth) * delta_t
    beta_init = (1 - indices / depth) * delta_t
    return np.concatenate([gamma_init, beta_init])


def bitstring_to_binary(bitstring: str) -> np.ndarray:
    """Convert a measurement bitstring to a binary solution vector.

    Convention: qubit i in |0> -> x_i=0, qubit i in |1> -> x_i=1.
    """
    return np.array([int(b) for b in bitstring], dtype=np.int64)


def most_probable_key(counts: dict[str, int], fallback: str) -> str:
    """Return the key with the highest count, or *fallback* if *counts* is empty."""
    if not counts:
        return fallback
    return max(counts, key=lambda k: counts[k])


def is_power_of_two(value: int) -> bool:
    """Return True when *value* is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0
