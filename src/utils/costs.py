"""Utility functions for calculating costs of problem instances."""

from __future__ import annotations

import numpy as np

from instance_gen_process.models import ProblemInstance, ProblemQUBO, ProblemTQUDO


def calculate_qubo_cost(problem: ProblemQUBO, solution: np.ndarray) -> float:
    """Calculate the QUBO cost of a solution (x^T Q x).

    This is the objective value in the QUBO formulation, which includes
    both the real cost terms and the penalty terms for constraint violations.

    Args:
        problem: The QUBO problem with QUBO_matrix.
        solution: Binary solution vector of shape (n_vars,) or (n_vars, 1).

    Returns:
        The QUBO cost value for the given solution.
    """
    x = np.asarray(solution).flatten()
    return float(x @ problem.QUBO_matrix @ x)


def calculate_tqudo_cost(
    problem: ProblemTQUDO,
    solution: np.ndarray,
) -> float:
    """Calculate the TQUDO cost of a solution.

    This is the objective value in the Tensor-QUDO formulation.
    Implementation pending: requires the exact solution format for TQUDO.

    Args:
        problem: The TQUDO problem with Etab and Ettprimeab tensors.
        solution: Solution tensor/vector (format TBD).

    Returns:
        The TQUDO cost value for the given solution.
    """
    x = np.asarray(solution).flatten()
    cost = 0
    for t, origin in enumerate(x[:-1]):
        destiny = x[t+1]
        cost += problem.Etab[t, origin, destiny]
        for tp, destiny in enumerate(x[t+1:]):
            t2 = t + tp
            cost += problem.Ettprimeab[t, t2, origin, destiny]

    raise cost


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
    travel_cost = sum(
        problem.prices_travels[t, sequence[t], sequence[t + 1]]
        for t in range(n_available - 1)
    ) + problem.prices_travels[n_available, sequence[n_available-1], n_available]


    return float(hotel_cost + travel_cost)