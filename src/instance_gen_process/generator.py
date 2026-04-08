"""Instance generation utilities for baseline experimentation."""

from __future__ import annotations

import logging
import random

import numpy as np

from instance_gen_process.models import (
    InstanceConfig,
    ProblemInstance,
    ProblemQUBO,
    ProblemTQUDO,
    RestrictionConfig,
)
from utils.constraints import idx, would_create_cycle

logger = logging.getLogger(__name__)


def generate_random_set_instances(
    config: InstanceConfig, n_instances: int, seed: int = 42
) -> list[ProblemInstance]:
    """Generate a set of random ProblemInstance from InstanceConfig ranges.

    Each instance is assigned a unique seed drawn from a master RNG seeded with
    ``seed``.  The per-instance seed is stored in ``ProblemInstance.seed`` so
    any individual instance can be reproduced without regenerating the others::

        instance = generate_random_instance(config, instance.seed)

    Args:
        config: Instance generation configuration (n_cities, price ranges, etc.).
        n_instances: Number of random instances to generate.
        seed: Master random seed that drives per-instance seed generation.

    Returns:
        List of n_instances ProblemInstance with valid precedences and price matrices.

    """
    master_rng = random.Random(seed)
    instance_seeds = [master_rng.randint(0, 2**32 - 1) for _ in range(n_instances)]
    return [generate_random_instance(config, s) for s in instance_seeds]


def generate_random_instance(config: InstanceConfig, seed: int) -> ProblemInstance:
    """Generate a single random ProblemInstance from InstanceConfig ranges.

    The ``seed`` is stored inside the returned ``ProblemInstance`` so the
    instance can be reproduced exactly by calling this function again with the
    same ``config`` and ``seed``.

    Args:
        config: Instance generation configuration.
        seed: Integer seed for this specific instance.

    Returns:
        ProblemInstance with acyclic precedences and random price matrices.

    """
    rng = random.Random(seed)

    n_cities = config.n_cities
    n_available = n_cities - 1
    precedences: list[tuple[int, int]] = []
    n_precedences = rng.randint(
        config.n_precedences_range[0],
        config.n_precedences_range[1],
    )
    attempts = 0
    max_attempts = (
        n_precedences * 20
    )  # Avoid infinite loop when no more valid precedences can be added
    while len(precedences) < n_precedences and attempts < max_attempts:
        origin = rng.randrange(n_available)
        available_destinations = [i for i in range(n_available) if i != origin]
        destination = rng.choice(available_destinations)
        if (origin, destination) not in precedences and not would_create_cycle(
            precedences, origin, destination
        ):
            precedences.append((origin, destination))
        attempts += 1

    if len(precedences) < n_precedences:
        logger.warning(
            "Could only generate %d of %d requested precedences after %d attempts "
            "(n_cities=%d). The acyclic constraint may limit the feasible set.",
            len(precedences),
            n_precedences,
            max_attempts,
            n_cities,
        )

    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    prices_hotels = np_rng.uniform(
        config.prices_range_hotels[0],
        config.prices_range_hotels[1],
        size=(n_available, n_available),
    )
    prices_travels = np_rng.uniform(
        config.prices_range_travels[0],
        config.prices_range_travels[1],
        size=(n_cities, n_cities, n_cities),
    )
    for i in range(n_cities):
        prices_travels[:, i, i] = 0

    return ProblemInstance(
        n_cities=n_cities,
        precedences=tuple(precedences),
        prices_hotels=prices_hotels,
        prices_travels=prices_travels,
        seed=seed,
    )


def generate_TQUDO_from_problem(
    problem: ProblemInstance, restriction: RestrictionConfig
) -> ProblemTQUDO:
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
    # Shape (n_available, d, d) where d = n_available.  The first dimension
    # doubles as the qudit count (n_qudits = n_available) even though only
    # indices 0..n_available-2 carry cost data — the last slice is all-zero
    # padding so that downstream code can infer n_qudits from Etab.shape[0].
    Etab = np.zeros((n_available, n_available, n_available), dtype=float)

    for t in range(n_available - 1):
        for origin in range(n_available):
            for destination in range(n_available):
                Etab[t, origin, destination] += problem.prices_travels[t + 1, origin, destination]
                Etab[t, origin, destination] += problem.prices_hotels[t, origin]
                if t == 0:
                    Etab[t, origin, destination] += problem.prices_travels[
                        0, n_available, origin
                    ]  # Closed loop
                if t == n_available - 2:
                    Etab[t, origin, destination] += problem.prices_travels[
                        n_available, destination, n_available
                    ]  # Closed loop
                    Etab[t, origin, destination] += problem.prices_hotels[
                        n_available - 1, destination
                    ]

    # Same first-dimension convention as Etab: shape (n_available, n_available, d, d)
    # so that Ettprimeab.shape[:2] matches Etab.shape[0] for n_qudits.
    #
    # NOTE: λ₀ (one-city-per-timestep) does NOT appear in Ettprimeab because
    # the qudit encoding inherently enforces exactly one city per timestep —
    # each qudit can only take a single value in {0, …, d-1}.  Only λ₁
    # (one-timestep-per-city, i.e. no duplicate cities) and λ₂ (precedence)
    # require explicit penalty terms.  In contrast, the QUBO formulation
    # uses binary one-hot encoding and needs λ₀ to penalize multiple active
    # bits per timestep.
    Ettprimeab = np.zeros((n_available, n_available, n_available, n_available), dtype=float)
    for t in range(n_available - 1):
        for t_prime in range(t + 1, n_available):
            for origin in range(n_available):
                Ettprimeab[t, t_prime, origin, origin] += restriction.lambda_1
                for destination in range(n_available):
                    for precedence in problem.precedences:
                        if origin == precedence[1] and destination == precedence[0]:
                            Ettprimeab[t, t_prime, origin, destination] += restriction.lambda_2

    max_etab = float(np.abs(Etab).max())
    max_ett = float(np.abs(Ettprimeab).max()) if Ettprimeab.size > 0 else 0.0
    energy_scale = float(max(max_etab, max_ett, 1.0))
    return ProblemTQUDO(
        Etab=Etab / energy_scale, Ettprimeab=Ettprimeab / energy_scale, energy_scale=energy_scale
    )


def generate_QUBO_from_problem(
    problem: ProblemInstance, restriction: RestrictionConfig
) -> ProblemQUBO:
    """Build the QUBO formulation from a ProblemInstance.

    Encodes costs and constraint penalties into a quadratic matrix for x^T Q x.
    See docs/formulations.md for the cost equations.

    Note:
        For any feasible permutation, the QUBO objective includes a constant
        offset from one-hot penalty linear terms:
        ``QUBO_cost = real_cost - (lambda_0 + lambda_1) * n_available``.
        The TQUDO objective matches ``real_cost`` for feasible solutions, so raw
        objectives are not directly comparable across formulations. Use
        ``utils.costs.calculate_real_cost`` for fair comparison.

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
    for t in range(n_available - 1):
        for i in range(n_available):
            idx_ti = idx(t, i, n_available)
            qubo_matrix[idx_ti, idx_ti] += problem.prices_hotels[t, i]
            if t == 0:
                qubo_matrix[idx_ti, idx_ti] += problem.prices_travels[0, n_available, i]
            if t == n_available - 2:
                idx_tp1_i = idx(t + 1, i, n_available)
                qubo_matrix[idx_tp1_i, idx_tp1_i] += problem.prices_travels[
                    n_available, i, n_available
                ]
                qubo_matrix[idx_tp1_i, idx_tp1_i] += problem.prices_hotels[n_available - 1, i]

            for j in range(n_available):
                idx_tp1_j = idx(t + 1, j, n_available)
                qubo_matrix[idx_ti, idx_tp1_j] += problem.prices_travels[t + 1, i, j] / 2
                qubo_matrix[idx_tp1_j, idx_ti] += (
                    problem.prices_travels[t + 1, i, j] / 2
                )  # For symmetry

    for t in range(n_available):
        for i in range(n_available):
            idx_ti = idx(t, i, n_available)
            qubo_matrix[idx_ti, idx_ti] -= restriction.lambda_0
            qubo_matrix[idx_ti, idx_ti] -= restriction.lambda_1
            for j in range(n_available):
                if i != j:
                    idx_tj = idx(t, j, n_available)
                    qubo_matrix[idx_ti, idx_tj] += restriction.lambda_0
            for t_prime in range(n_available):
                if t != t_prime:
                    idx_tp_i = idx(t_prime, i, n_available)
                    qubo_matrix[idx_ti, idx_tp_i] += restriction.lambda_1

        for t_prime in range(t + 1, n_available):
            for precedence in problem.precedences:
                i, j = precedence
                idx_tp_i = idx(t_prime, i, n_available)
                idx_tj = idx(t, j, n_available)
                qubo_matrix[idx_tp_i, idx_tj] += restriction.lambda_2 / 2
                qubo_matrix[idx_tj, idx_tp_i] += restriction.lambda_2 / 2  # Symmetry

    energy_scale = float(max(float(np.abs(qubo_matrix).max()), 1.0))
    return ProblemQUBO(qubo_matrix=qubo_matrix / energy_scale, energy_scale=energy_scale)
