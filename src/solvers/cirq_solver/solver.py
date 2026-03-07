"""Cirq-based backend scaffold for QAOA variants."""

from __future__ import annotations

from instance_gen_process import ProblemInstance
from solvers.base import SolverResult, SolverRunConfig


class CirqSolver:
    """Placeholder Cirq solver for future qudit and qubit QAOA experiments."""

    solver_name = "cirq"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        _ = (instance, run_config)
        raise NotImplementedError("Cirq solver is scaffolded but not implemented yet.")

