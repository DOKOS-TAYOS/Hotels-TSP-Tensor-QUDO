"""Solver interfaces and backend scaffolds."""

from solvers.base import SolverProtocol, SolverResult, SolverRunConfig
from solvers.cirq_solver import CirqSolver
from solvers.cudaq_solver import CudaqSolver
from solvers.simulated_annealing import SimulatedAnnealingSolver

__all__ = [
    "CirqSolver",
    "CudaqSolver",
    "SimulatedAnnealingSolver",
    "SolverProtocol",
    "SolverResult",
    "SolverRunConfig",
]

