"""Simulated annealing backend scaffold."""

from __future__ import annotations

from instance_gen_process import ProblemInstance
from solvers.base import SolverResult, SolverRunConfig


class SimulatedAnnealingSolver:
    """Placeholder solver that will host the SA implementation."""

    solver_name = "simulated_annealing"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        _ = (instance, run_config)
        raise NotImplementedError(
            "Simulated annealing solver is scaffolded but not implemented yet."
        )

