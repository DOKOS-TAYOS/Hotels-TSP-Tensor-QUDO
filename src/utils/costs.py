"""Utility functions for calculating costs of problem instances."""

from __future__ import annotations

import numpy as np

from instance_gen_process.models import ProblemInstance, ProblemQUBO, ProblemTQUDO


def calculate_qubo_cost_from_sequence(
    problem: ProblemQUBO,
    sequence: np.ndarray,
    n_available: int,
) -> float:
    """QUBO cost ``x^T Q x`` for a route without materialising the binary vector.

    For each timestep ``t`` exactly one bit is 1 at
    ``idx[t] = t * n_available + sequence[t]``. The energy is the submatrix sum
    ``sum_{t,t'} Q[idx[t], idx[t']]``, i.e. O(n_available^2) in the route
    length rather than O(n_vars^2) over the full ``n_vars``-dimensional Q.

    Args:
        problem: QUBO problem with ``qubo_matrix`` and ``energy_scale``.
        sequence: Route ``sequence[t]`` = city index at timestep ``t``.
        n_available: Number of non-depot cities (``n_cities - 1``).

    Returns:
        The QUBO objective in original problem units (same as
        :func:`calculate_qubo_cost` for the equivalent binary vector).

    """
    seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
    t = np.arange(n_available, dtype=np.int64)
    idx = t * n_available + seq
    q = np.asarray(problem.qubo_matrix, dtype=np.float64)
    return float(np.sum(q[np.ix_(idx, idx)])) * problem.energy_scale


def calculate_qubo_cost(problem: ProblemQUBO, solution: np.ndarray) -> float:
    """Calculate the QUBO cost of a solution (x^T Q x) in original problem units.

    This is the objective value in the QUBO formulation, which includes
    both the real cost terms and the penalty terms for constraint violations.
    See docs/formulations.md for the cost equations.

    Note:
        For feasible solutions the QUBO cost differs from the real travel and
        hotel cost by a constant offset:
        ``QUBO_cost = real_cost - (lambda_0 + lambda_1) * n_available``.
        Use :func:`calculate_real_cost` for formulation-independent comparisons.

    The stored ``qubo_matrix`` may be normalised; this function always returns
    the cost rescaled to original problem units via ``problem.energy_scale``.

    Args:
        problem: The QUBO problem with qubo_matrix (potentially normalised).
        solution: Binary solution vector of shape (n_vars,) or (n_vars, 1).

    Returns:
        The QUBO cost value for the given solution in original problem units.

    """
    x = np.asarray(solution).flatten()
    return float(x @ problem.qubo_matrix @ x) * problem.energy_scale


def calculate_tqudo_cost(
    problem: ProblemTQUDO,
    solution: np.ndarray,
) -> float:
    """Calculate the TQUDO cost of a solution in original problem units.

    This is the objective value in the Tensor-QUDO formulation.
    For feasible solutions (valid permutations) this equals the real
    travel+hotel cost directly, unlike the QUBO formulation which
    includes a constant penalty offset (see :func:`calculate_qubo_cost`).
    See docs/formulations.md for the cost equations.

    The stored tensors may be normalised; this function always returns the
    cost rescaled to original problem units via ``problem.energy_scale``.

    Equivalent loop-based form (for reference)::

        cost = 0.0
        for t, origin in enumerate(x[:-1]):
            destination = x[t + 1]
            cost += Etab[t, origin, destination]
            for tp, dest_tp in enumerate(x[t + 1:]):
                t_prime = t + 1 + tp
                cost += Ettprimeab[t, t_prime, origin, dest_tp]

    Args:
        problem: The TQUDO problem with Etab and Ettprimeab tensors
            (potentially normalised).
        solution: Qudit sequence of city indices, shape (n_qudits,).

    Returns:
        The TQUDO cost value for the given solution in original problem units.

    """
    x = np.asarray(solution, dtype=int).flatten()
    n = len(x)

    ts = np.arange(n - 1)
    etab_cost = float(np.sum(problem.Etab[ts, x[:-1], x[1:]]))

    t_left, t_right = np.triu_indices(n, k=1)
    ett_cost = float(np.sum(
        problem.Ettprimeab[t_left, t_right, x[t_left], x[t_right]]
    ))

    return (etab_cost + ett_cost) * problem.energy_scale


def calculate_real_cost(problem: ProblemInstance, sequence: list[int]) -> float:
    """Calculate the real cost of a route, assuming constraints are satisfied.

    Sums hotel costs (per timestep) and travel costs (between consecutive steps).
    Does not validate that the sequence satisfies precedence constraints.

    Args:
        problem: The problem instance with prices_hotels and prices_travels.
        sequence: Route as list of city indices, sequence[t] = city at time t.
                  Must have length n_cities - 1.

    Returns:
        Total cost: sum of hotel costs + sum of travel costs.

    """
    n_available = problem.n_cities - 1
    if len(sequence) != n_available:
        raise ValueError(
            f"Sequence length {len(sequence)} must equal n_available={n_available}"
        )

    hotel_cost = sum(
        problem.prices_hotels[t, sequence[t]] for t in range(n_available)
    )
    travel_cost = (
        problem.prices_travels[0, n_available, sequence[0]]  # start -> first
        + sum(
            problem.prices_travels[t + 1, sequence[t], sequence[t + 1]]
            for t in range(n_available - 1)
        )
        + problem.prices_travels[n_available, sequence[n_available - 1], n_available]  # last -> start
    )
    return float(hotel_cost + travel_cost)