"""Instance generation utilities for baseline experimentation."""

from __future__ import annotations

import random
import numpy as np

from utils.constraints import idx
from instance_gen_process.models import InstanceConfig, ProblemInstance, ProblemQUBO, ProblemTQUDO, RestrictionConfig

def generate_random_set_instances(config: InstanceConfig, n_instances: int, seed: int=42) -> list[ProblemInstance]:
    """Generate a set of random `ProblemInstance` from `InstanceConfig` ranges."""
    rng = random.Random(seed)
    problems_list = []
    for _ in range(n_instances):
        problems_list.append(generate_random_instance(config, rng))

    return problems_list


def _would_create_cycle(precedences: list[tuple[int, int]], origin: int, destiny: int) -> bool:
    """True if adding (origin, destiny) would create a cycle.
    A cycle occurs when destiny can already reach origin (directly or via other rules)."""
    # Build adjacency: (a, b) means a -> b. We need to check if destiny ->* origin.
    adj: dict[int, list[int]] = {}
    for a, b in precedences:
        adj.setdefault(a, []).append(b)
    # BFS from destiny: can we reach origin?
    seen: set[int] = {destiny}
    queue: list[int] = [destiny]
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
    """Generate a random `ProblemInstance` from `InstanceConfig` ranges."""

    n_cities = config.n_cities
    n_available = n_cities - 1
    precedences: list[tuple[int, int]] = []
    n_precedences = rng.randint(
                        config.n_precedences_range[0],
                        config.n_precedences_range[1]
                        )
    attempts = 0
    max_attempts = n_precedences * 20  # Evitar bucle infinito si es imposible añadir más
    while len(precedences) < n_precedences and attempts < max_attempts:
        origin = rng.randint(n_available)
        available_destinations = [i for i in range(n_available) if i != origin]
        destiny = rng.choice(available_destinations)
        if not _would_create_cycle(precedences, origin, destiny):
            precedences.append((origin, destiny))
        attempts += 1

    prices_hotels = np.random.uniform(
        config.prices_range_hotels[0],
        config.prices_range_hotels[1],
        size=(n_available, n_available)
    )
    prices_travels = np.random.uniform(
        config.prices_range_travels[0],
        config.prices_range_travels[1],
        size=(n_available, n_available, n_available)
    )
    # Set diagonals of the last two indices to 0
    for i in range(n_available):
        prices_travels[:, i, i] = 0

    return ProblemInstance(n_cities=n_cities, precedences=precedences, prices_hotels=prices_hotels, prices_travels=prices_travels)


def generate_TQUDO_from_problem(problem: ProblemInstance, restriction: RestrictionConfig) -> ProblemTQUDO:
    """Generates the Tensor QUDO of the Problem.
    """
    n_cities = problem.n_cities
    n_available = n_cities - 1
    Etab = problem.prices_travels

    for t in range(n_available):
        for origin in range(n_available):
            for destiny in range(n_available):
                Etab[t, origin, destiny] += problem.prices_hotels[t, origin]
    
    Ettprimeab = np.zeros((n_available, n_available, n_available, n_available), dtype=float)
    for t in range(n_available):
        for t_prime in range(t+1, n_available):
            for origin in range(n_available):
                Ettprimeab[t, t_prime, origin, origin] += restriction.lambda_1
                for destiny in range(n_available):
                    for precedence in problem.precedences:
                        if origin == precedence[1] and destiny == precedence[0]:
                            Ettprimeab[t, t_prime, origin, destiny] += restriction.lambda_2
    
    return ProblemTQUDO(Etab=Etab, Ettprimeab=Ettprimeab)


def generate_QUBO_from_problem(problem: ProblemInstance, restriction: RestrictionConfig) -> ProblemQUBO:
    """Generates the QUBO of the Problem.
    """
    n_cities = problem.n_cities
    n_available = n_cities - 1

    qubo_matrix = np.zeros((n_available, n_available), dtype=float)
    for t in range(n_available):
        for i in range(n_available):
            first_index = idx(t, i, n_available)
            qubo_matrix[first_index, first_index] += problem.prices_hotels[t,i]
            for t2 in range(n_available):
                second_index_t2 = idx(t2,i)
                if t!=t2:
                    qubo_matrix[first_index, second_index_t2] += restriction.lambda_1
            for j in range(n_available):
                second_index_j = idx(t,j)
                second_index_tp1_j = idx(t+1,j)
                if i!=j:
                    qubo_matrix[first_index, second_index_tp1_j] += problem.prices_travels[t,i,j]
                    qubo_matrix[first_index, second_index_j] += restriction.lambda_0

    
    for t in range(n_available):
        for t2 in range(t+1,n_available):
            for precedence in problem.precedences:
                i, j = precedence
                second_index_t2 = idx(t2,i)
                second_index_j = idx(t,j)
                qubo_matrix[second_index_t2, second_index_j] += restriction.lambda_2
                        
    return ProblemQUBO(QUBO_matrix=qubo_matrix)


    


