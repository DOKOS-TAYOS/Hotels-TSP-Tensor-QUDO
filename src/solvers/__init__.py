"""Solver interfaces and backend scaffolds.

Backend solver classes (CirqSolver, CudaqSolver, SimulatedAnnealingSolver)
are imported lazily so that loading this package does not pull in heavy
optional dependencies (e.g. ``cirq``, ``cudaq``) that may not be installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from solvers.base import SolverProtocol, SolverResult, SolverRunConfig
from solvers.noise import NoiseConfig

if TYPE_CHECKING:
    from solvers.cirq_solver import CirqSolver as CirqSolver
    from solvers.cudaq_solver import CudaqSolver as CudaqSolver
    from solvers.simulated_annealing import SimulatedAnnealingSolver as SimulatedAnnealingSolver

__all__ = [
    "CirqSolver",
    "CudaqSolver",
    "NoiseConfig",
    "SimulatedAnnealingSolver",
    "SolverProtocol",
    "SolverResult",
    "SolverRunConfig",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CirqSolver": ("solvers.cirq_solver", "CirqSolver"),
    "CudaqSolver": ("solvers.cudaq_solver", "CudaqSolver"),
    "SimulatedAnnealingSolver": ("solvers.simulated_annealing", "SimulatedAnnealingSolver"),
}


def __getattr__(name: str) -> object:
    """Lazily import ``CirqSolver``, ``CudaqSolver``, or ``SimulatedAnnealingSolver``.

    Args:
        name: Attribute requested on ``solvers``.

    Returns:
        The solver class object.

    Raises:
        AttributeError: If ``name`` is not a known lazy export.

    """
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
