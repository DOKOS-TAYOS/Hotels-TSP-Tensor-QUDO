"""Data models for generated travel routing (Hotel TSP) instances."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class InstanceConfig:
    """Configuration that controls random instance generation.

    Attributes:
        n_cities: Total cities including the depot.
        n_precedences_range: Inclusive min/max count of random precedence edges.
        prices_range_hotels: Uniform sampling range for hotel prices.
        prices_range_travels: Uniform sampling range for travel prices.
        seed: Master seed for the workflow when generating instance batches.

    """

    n_cities: int
    n_precedences_range: tuple[int, int]
    prices_range_hotels: tuple[float, float]
    prices_range_travels: tuple[float, float]
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ProblemInstance:
    """Canonical in-memory problem representation consumed by solvers.

    Attributes:
        n_cities: Total number of cities (including the depot).
        precedences: List of (origin, destination) precedence pairs.
        prices_hotels: 2D array of hotel prices (n_available x n_available).
        prices_travels: 3D array of travel prices (n_cities x n_cities x n_cities).
        seed: Integer seed used to generate this instance. Store it to allow
            exact reproduction via ``generate_random_instance(config, seed)``.

    """

    n_cities: int
    precedences: tuple[tuple[int, int], ...]
    prices_hotels: np.ndarray  # 2 dimensions
    prices_travels: np.ndarray  # 3 dimensions
    seed: int = 0

@dataclass(frozen=True, slots=True)
class ProblemTQUDO:
    """Tensor-QUDO formulation: Hamiltonian terms for quantum device.

    Cost: C(x) = sum_t E_{t,x_t,x_{t+1}} + sum_{t,t'>t} E_{t,t',x_t,x_{t'}}.
    See docs/formulations.md for full equations.

    Attributes:
        Etab: 3D tensor (t, origin, destination) for travel and hotel costs.
            Normalised so that max(|Etab|, |Ettprimeab|) == 1.
        Ettprimeab: 4D tensor (t, t_prime, origin, destination) for penalties.
            Normalised together with Etab.
        energy_scale: Factor by which the original tensors were divided during
            normalisation.  Multiply any sampled cost by this value to recover
            the original-units objective.

    """

    Etab: np.ndarray  # 3 dimensions
    Ettprimeab: np.ndarray  # 4 dimensions
    energy_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class ProblemQUBO:
    """QUBO formulation: quadratic matrix for quantum/classical solvers.

    Cost: C(x) = x^T Q x = sum_i Q_ii x_i + sum_{i<j} 2 Q_ij x_i x_j.
    See docs/formulations.md for full equations.

    Attributes:
        qubo_matrix: Symmetric normalised matrix of shape (n_vars, n_vars)
            where n_vars = n_available^2.  All entries are in [-1, 1].
        energy_scale: Factor by which the original matrix was divided during
            normalisation.  Multiply any sampled cost by this value to recover
            the original-units objective.

    """

    qubo_matrix: np.ndarray  # 2 dimensions
    energy_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class RestrictionConfig:
    """Penalty coefficients for QUBO/TQUDO constraint encoding.

    Attributes:
        lambda_0: Penalty for "not exactly one node per timestep".
        lambda_1: Penalty for "not exactly one timestep per node".
        lambda_2: Penalty for precedence constraint violations.

    """

    lambda_0: float
    lambda_1: float
    lambda_2: float