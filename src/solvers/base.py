"""Common solver protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from instance_gen_process.models import ProblemInstance, RestrictionConfig

if TYPE_CHECKING:
    from solvers.noise import NoiseConfig


OptimizerType = Literal["COBYLA", "Powell", "L-BFGS-B", "SLSQP", "Nelder-Mead"]


@dataclass(frozen=True, slots=True)
class SolverRunConfig:
    """Generic run controls shared across solver backends."""

    max_iterations: int = 1_000
    timeout_seconds: float | None = None
    formulation: Literal["tqudo", "qubo"] = "tqudo"
    restriction_config: RestrictionConfig | None = None
    qaoa_depth: int = 1
    qaoa_max_iter: int = 100
    # Shots per objective evaluation for all sampling-based QAOA backends
    # (both TQUDO and QUBO formulations).
    qaoa_shots: int = 500
    # Shots used to sample the final candidate solution for any QAOA backend.
    qaoa_sample_shots: int = 1000
    seed: int | None = None
    optimizer: OptimizerType = "COBYLA"
    delta_t: float = 0.55
    # Noise simulation (optional, disabled by default)
    noise_config: NoiseConfig = field(default_factory=lambda: __import__('solvers.noise', fromlist=['NoiseConfig']).NoiseConfig())
    # Simulated annealing parameters
    sa_t_initial: float = 1000.0    # Initial temperature
    sa_t_final: float = 1e-6        # Final (minimum) temperature
    sa_alpha: float = 0.995         # Geometric cooling factor (T *= alpha each step)


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Standardized solver output.

    metadata may include: initial_energy (float), energy_history (list[float]),
    best_sequence, best_bitstring, best_binary, real_cost, etc.
    """

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
