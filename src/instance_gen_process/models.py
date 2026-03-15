"""Data models for generated aircraft loading instances."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class InstanceConfig:
    """Configuration that controls random instance generation."""

    n_cities: int
    n_precedences_range: tuple[int, int]
    prices_range_hotels: tuple[float, float]
    prices_range_travels: tuple[float, float]
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ProblemInstance:
    """Canonical in-memory problem representation consumed by solvers."""

    n_cities: int
    precedences: list[tuple[int, int]]
    prices_hotels: np.ndarray # 2 dimensions
    prices_travels: np.ndarray # 3 dimensions

@dataclass(frozen=True, slots=True)
class ProblemTQUDO:
    """
    Representation for the quantum device hamiltonian terms.
    """
    
    Etab: np.ndarray # 3 dimensions
    Ettprimeab: np.ndarray # 4 dimensions

@dataclass(frozen=True, slots=True)
class ProblemTQUDO:
    """
    Representation for the quantum device hamiltonian terms.
    """
    
    Etab: np.ndarray # 3 dimensions
    Ettprimeab: np.ndarray # 4 dimensions

@dataclass(frozen=True, slots=True)
class ProblemQUBO:
    """
    Representation for the quantum device hamiltonian terms.
    """
    
    QUBO_matrix: np.ndarray # 2 dimensions

@dataclass(frozen=True, slots=True)
class RestrictionConfig:
    """
    Class for the restriction terms.
    Lambda0 is the non two nodes per time
    Lambda1 is the non two times per node
    Lambda2 is the precedence restriction
    """
    lambda_0: float
    lambda_1: float
    lambda_2: float