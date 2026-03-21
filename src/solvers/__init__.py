"""Solver interfaces and backend scaffolds."""

from solvers.base import SolverProtocol, SolverResult, SolverRunConfig
from solvers.cirq_solver import CirqSolver
from solvers.cudaq_solver import CudaqSolver
from solvers.noise import NoiseConfig
from solvers.simulated_annealing import SimulatedAnnealingSolver

__all__ = [
    "CirqSolver",
    "CudaqSolver",
    "NoiseConfig",
    "SimulatedAnnealingSolver",
    "SolverProtocol",
    "SolverResult",
    "SolverRunConfig",
]

