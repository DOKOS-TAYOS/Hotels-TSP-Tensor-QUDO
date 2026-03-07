"""CUDA-Q backend scaffold for QAOA and noisy simulation experiments."""

from __future__ import annotations

from instance_gen_process import ProblemInstance
from solvers.base import SolverResult, SolverRunConfig


class CudaqSolver:
    """Placeholder CUDA-Q solver."""

    solver_name = "cudaq"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        _ = (instance, run_config)
        raise NotImplementedError("CUDA-Q solver is scaffolded but not implemented yet.")

