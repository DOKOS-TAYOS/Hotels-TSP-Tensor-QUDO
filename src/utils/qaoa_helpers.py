"""Shared helpers used across QAOA circuit implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def tqa_init_params(depth: int, delta_t: float) -> np.ndarray:
    """Return TQA (Trotterized Quantum Annealing) initial QAOA angles.

    Uses ``gamma_i = (i / p) * delta_t`` and ``beta_i = (1 - i / p) * delta_t``
    for ``i = 1, …, p``.

    Args:
        depth: QAOA layers p.
        delta_t: Overall scale for initial ``gamma`` and ``beta`` ramps.

    Returns:
        1-D array of shape ``(2 * depth,)``:
        ``[gamma_1, …, gamma_p, beta_1, …, beta_p]``.
    """
    indices = np.arange(1, depth + 1)
    gamma_init = (indices / depth) * delta_t
    beta_init = (1 - indices / depth) * delta_t
    return np.concatenate([gamma_init, beta_init])


def bitstring_to_binary(bitstring: str) -> np.ndarray:
    """Convert a measurement bitstring to a binary solution vector.

    Args:
        bitstring: Characters ``'0'`` and ``'1'`` in qubit order.

    Returns:
        Integer array with ``x_i = 0`` for ``|0⟩`` and ``x_i = 1`` for ``|1⟩``.
    """
    return np.array([int(b) for b in bitstring], dtype=np.int64)


def measurement_histogram_for_json(
    samples: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    """Convert backend shot histogram to JSON-friendly dict sorted by descending count.

    Args:
        samples: Mapping from measurement key to count, or None.

    Returns:
        ``{key: int(count)}`` sorted by count descending, or None when *samples* is None.
    """
    if samples is None:
        return None
    counts: dict[str, int] = {str(k): int(v) for k, v in samples.items()}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def most_probable_key(counts: dict[str, int], fallback: str) -> str:
    """Return the measurement key with the largest count.

    Args:
        counts: Histogram of bitstrings or qudit keys to counts.
        fallback: Value returned when *counts* is empty.

    Returns:
        Argmax key by count, or *fallback*.
    """
    if not counts:
        return fallback
    return max(counts, key=lambda k: counts[k])


def is_power_of_two(value: int) -> bool:
    """Return whether *value* is a positive integer power of two.

    Args:
        value: Integer to test.

    Returns:
        True iff ``value > 0`` and only one bit is set in binary.
    """
    return value > 0 and (value & (value - 1)) == 0
