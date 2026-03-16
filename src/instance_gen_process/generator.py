"""Instance generation utilities for baseline experimentation."""

from __future__ import annotations

import random
import numpy as np

from utils.constraints import idx
from instance_gen_process.models import InstanceConfig, ProblemInstance, ProblemQUBO, ProblemTQUDO, RestrictionConfig

def generate_random_set_instances(config: InstanceConfig, n_instances: int, seed: int = 42) -> list[ProblemInstance]:
    """Generate a set of random ProblemInstance from InstanceConfig ranges.

    Args:
        config: Instance generation configuration (n_cities, price ranges, etc.).
        n_instances: Number of random instances to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of n_instances ProblemInstance with valid precedences and price matrices.
    """
    rng = random.Random(seed)
    problem_instances = []
    for _ in range(n_instances):
        problem_instances.append(generate_random_instance(config, rng))

    return problem_instances


def _would_create_cycle(precedences: list[tuple[int, int]], origin: int, destination: int) -> bool:
    """Check if adding (origin, destination) would create a cycle in the precedence graph.

    A cycle occurs when destination can already reach origin (directly or via other rules).
    Uses BFS from destination to detect reachability to origin.

    Args:
        precedences: Existing precedence edges (a, b) meaning a must precede b.
        origin: Proposed edge origin.
        destination: Proposed edge destination.

    Returns:
        True if adding the edge would create a cycle, False otherwise.
    """
    # Build adjacency: (a, b) means a -> b. We need to check if destination ->* origin.
    adj: dict[int, list[int]] = {}
    for a, b in precedences:
        adj.setdefault(a, []).append(b)
    # BFS from destination: can we reach origin?
    seen: set[int] = {destination}
    queue: list[int] = [destination]
    while queue:
        node = queue.pop(0)
        for neighbor in adj.get(node, []):
            if neighbor == origin:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return False


def generate_random_instance(config: InstanceConfig, rng: random.Random) -> ProblemInstance:
    """Generate a single random ProblemInstance from InstanceConfig ranges.

    Args:
        config: Instance generation configuration.
        rng: Seeded random generator for reproducibility.

    Returns:
        ProblemInstance with acyclic precedences and random price matrices.
    """

    n_cities = config.n_cities
    n_available = n_cities - 1
    precedences: list[tuple[int, int]] = []
    n_precedences = rng.randint(
                        config.n_precedences_range[0],
                        config.n_precedences_range[1]
                        )
    attempts = 0
    max_attempts = n_precedences * 20  # Avoid infinite loop when no more valid precedences can be added
    while len(precedences) < n_precedences and attempts < max_attempts:
        origin = rng.randrange(n_available)
        available_destinations = [i for i in range(n_available) if i != origin]
        destination = rng.choice(available_destinations)
        if not _would_create_cycle(precedences, origin, destination):
            precedences.append((origin, destination))
        attempts += 1

    prices_hotels = np.random.uniform(
        config.prices_range_hotels[0],
        config.prices_range_hotels[1],
        size=(n_available, n_available)
    )
    prices_travels = np.random.uniform(
        config.prices_range_travels[0],
        config.prices_range_travels[1],
        size=(n_cities, n_cities, n_cities)
    )
    # Set diagonals of the last two indices to 0
    for i in range(n_cities):
        prices_travels[:, i, i] = 0

    return ProblemInstance(n_cities=n_cities, precedences=precedences, prices_hotels=prices_hotels, prices_travels=prices_travels)


def generate_TQUDO_from_problem(problem: ProblemInstance, restriction: RestrictionConfig) -> ProblemTQUDO:
    """Build the Tensor-QUDO formulation from a ProblemInstance.

    Encodes travel costs, hotel costs, and precedence penalties into Etab and Ettprimeab tensors.
    See docs/formulations.md for the cost equations.

    Args:
        problem: Canonical problem with precedences and price matrices.
        restriction: Penalty coefficients (lambda_0, lambda_1, lambda_2).

    Returns:
        ProblemTQUDO with Etab (3D) and Ettprimeab (4D) tensors.
    """
    n_cities = problem.n_cities
    n_available = n_cities - 1
    Etab = np.zeros((n_available, n_available, n_available), dtype=float)

    for t in range(n_available-1):
        for origin in range(n_available):
            for destination in range(n_available):
                Etab[t, origin, destination] += problem.prices_travels[t+1, origin, destination]
                Etab[t, origin, destination] += problem.prices_hotels[t, origin]
                if t == 0:
                    Etab[t, origin, destination] += problem.prices_travels[0, n_available, origin]  # Closed loop
                if t == n_available - 2:
                    Etab[t, origin, destination] += problem.prices_travels[n_available, destination, n_available]  # Closed loop
                    Etab[t, origin, destination] += problem.prices_hotels[n_available - 2, destination]
                
    # This may be changed to Tab if in the solver we take into acount that t'>t
    Ettprimeab = np.zeros((n_available, n_available, n_available, n_available), dtype=float)
    for t in range(n_available-1):
        for t_prime in range(t+1, n_available):
            for origin in range(n_available):
                Ettprimeab[t, t_prime, origin, origin] += restriction.lambda_1
                for destination in range(n_available):
                    for precedence in problem.precedences:
                        if origin == precedence[1] and destination == precedence[0]:
                            Ettprimeab[t, t_prime, origin, destination] += restriction.lambda_2
    
    return ProblemTQUDO(Etab=Etab, Ettprimeab=Ettprimeab)


def generate_QUBO_from_problem(problem: ProblemInstance, restriction: RestrictionConfig) -> ProblemQUBO:
    """Build the QUBO formulation from a ProblemInstance.

    Encodes costs and constraint penalties into a quadratic matrix for x^T Q x.
    See docs/formulations.md for the cost equations.

    Args:
        problem: Canonical problem with precedences and price matrices.
        restriction: Penalty coefficients (lambda_0, lambda_1, lambda_2).

    Returns:
        ProblemQUBO with qubo_matrix of shape (n_available^2, n_available^2).
    """
    n_cities = problem.n_cities
    n_available = n_cities - 1
    n_vars = n_available * n_available

    qubo_matrix = np.zeros((n_vars, n_vars), dtype=float)
    for t in range(n_available-1):
        for i in range(n_available):
            idx_ti = idx(t, i, n_available)
            qubo_matrix[idx_ti, idx_ti] += problem.prices_hotels[t, i]
            if t == 0:
                qubo_matrix[idx_ti, idx_ti] += problem.prices_travels[0, n_available, i]

            for j in range(n_available):
                idx_tj = idx(t, j, n_available)
                idx_tp1_j = idx(t + 1, j, n_available)
                if i != j:
                    qubo_matrix[idx_ti, idx_tp1_j] += problem.prices_travels[t+1, i, j] / 2
                    qubo_matrix[idx_tp1_j, idx_ti] += problem.prices_travels[t+1, i, j] / 2 # For symmetry
                    if t == n_available-2:
                        qubo_matrix[idx_tp1_j, idx_tp1_j] += problem.prices_travels[n_available, j, n_available]
                        qubo_matrix[idx_tp1_j, idx_tp1_j] += problem.prices_hotels[n_available-1, j]

    for t in range(n_available):
        for i in range(n_available):
            idx_ti = idx(t, i, n_available)
            for j in range(n_available):
                if i != j:
                    idx_tj = idx(t, j, n_available)
                    qubo_matrix[idx_ti, idx_tj] += restriction.lambda_0 / 2
            for t_prime in range(n_available):
                if t != t_prime:
                    idx_tp_i = idx(t_prime, i, n_available)
                    qubo_matrix[idx_ti, idx_tp_i] += restriction.lambda_1 / 2

        for t_prime in range(t+1, n_available):
            for precedence in problem.precedences:
                i, j = precedence
                idx_tp_i = idx(t_prime, i, n_available)
                idx_tj = idx(t, j, n_available)
                qubo_matrix[idx_tp_i, idx_tj] += restriction.lambda_2 / 2
                qubo_matrix[idx_tj, idx_tp_i] += restriction.lambda_2 / 2 # Symmetry
                        
    return ProblemQUBO(qubo_matrix=qubo_matrix)


    


