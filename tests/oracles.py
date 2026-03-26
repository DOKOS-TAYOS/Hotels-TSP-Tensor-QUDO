"""Reference implementations for tests (naive loops, independent of vectorised code)."""

from __future__ import annotations

import numpy as np


def tqudo_cost_naive_loops(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    seq: np.ndarray | list[int],
    energy_scale: float,
) -> float:
    """TQUDO objective from the literal double loops in calculate_tqudo_cost docstring.

    Independent of ``np.triu_indices`` / batch gather staging.
    """
    x = np.asarray(seq, dtype=int).flatten()
    n = int(x.shape[0])
    cost = 0.0
    for t in range(n - 1):
        origin = int(x[t])
        destination = int(x[t + 1])
        cost += float(Etab[t, origin, destination])
    for t in range(n - 1):
        origin = int(x[t])
        dest_slice = x[t + 1 :]
        for tp, dest_tp in enumerate(dest_slice):
            t_prime = t + 1 + tp
            cost += float(Ettprimeab[t, t_prime, origin, int(dest_tp)])
    return cost * float(energy_scale)
