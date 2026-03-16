"""Common solver protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from instance_gen_process import ProblemInstance
from instance_gen_process.models import RestrictionConfig


@dataclass(frozen=True, slots=True)
class SolverRunConfig:
    """Generic run controls shared across solver backends."""

    max_iterations: int = 1_000
    timeout_seconds: float | None = None
    # CUDA-Q QAOA specific (ignored by other solvers)
    formulation: Literal["tqudo", "qubo"] = "tqudo"
    restriction_config: RestrictionConfig | None = None
    qaoa_depth: int = 1
    qaoa_max_iter: int = 100
    qaoa_shots: int = 500
    qaoa_sample_shots: int = 1000
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Standardized solver output."""

    solver_name: str
    objective_value: float
    feasible: bool
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SolverProtocol(Protocol):
    """Contract expected from every backend implementation."""

    solver_name: str

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        """Execute optimization and return a standardized result.

        Args:
            instance: Problem to solve (precedences, prices).
            run_config: Solver controls (max_iterations, timeout).

        Returns:
            SolverResult with objective_value, feasible, runtime_seconds.
        """

