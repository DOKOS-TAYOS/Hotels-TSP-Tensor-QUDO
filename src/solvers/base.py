"""Common solver protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from instance_gen_process.models import ProblemInstance, RestrictionConfig

if TYPE_CHECKING:
    from solvers.noise import NoiseConfig


OptimizerType = Literal["COBYLA", "Powell", "L-BFGS-B", "SLSQP", "Nelder-Mead"]


def _default_noise_config() -> NoiseConfig:
    """Return default ``NoiseConfig`` with noise simulation disabled."""
    from solvers.noise import NoiseConfig

    return NoiseConfig()


@dataclass(frozen=True, slots=True)
class SolverRunConfig:
    """Generic run controls shared across solver backends.

    Attributes:
        max_iterations: Upper bound on solver iterations (where applicable).
        timeout_seconds: Optional wall-clock limit in seconds.
        formulation: ``qubo``, ``tqudo`` (native qudits), or ``tqudo_virtual``.
        restriction_config: QUBO/TQUDO penalty weights; defaults if None.
        qaoa_depth: QAOA circuit depth p.
        qaoa_max_iter: Classical optimizer iterations for QAOA angles.
        qaoa_shots: Shots per objective evaluation for sampling QAOA.
        qaoa_sample_shots: Shots when drawing final (and initial) samples.
        seed: Optional RNG seed.
        optimizer: SciPy ``minimize`` method name for QAOA.
        delta_t: TQA-style initial parameter scale.
        optimizer_tol: SciPy classical-optimizer stopping tolerance for QAOA.
        noise_config: Optional noise simulation parameters.
        sa_t_initial: Simulated annealing start temperature.
        sa_t_final: Simulated annealing end temperature.
        sa_alpha: Multiplicative cooling factor per SA step.
        brute_force_max_assignments_tqudo: Max TQUDO assignments (full space is ``n^n``,
            with ``n = n_cities - 1`` capped at 8).
        brute_force_max_assignments_qubo: Max QUBO assignments (full space is ``2^(n^2)``,
            with ``n^2`` binary vars capped at 30).

    """

    max_iterations: int = 1_000
    timeout_seconds: float | None = None
    formulation: Literal["qubo", "tqudo", "tqudo_virtual"] = "tqudo"
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
    optimizer_tol: float = 1e-6
    # Noise simulation (optional, disabled by default)
    noise_config: NoiseConfig = field(default_factory=_default_noise_config)
    # Simulated annealing parameters
    sa_t_initial: float = 1000.0  # Initial temperature
    sa_t_final: float = 1e-6  # Final (minimum) temperature
    sa_alpha: float = 0.995  # Geometric cooling factor (T *= alpha each step)
    # Brute-force enumeration caps (solver ``brute_force`` only; defaults = full allowed spaces)
    brute_force_max_assignments_tqudo: int = 8**8
    brute_force_max_assignments_qubo: int = 2**30


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Standardized solver output from any backend.

    Attributes:
        solver_name: Backend identifier (e.g. ``cirq``, ``cudaq``).
        objective_value: Best objective found (units depend on formulation).
        feasible: Whether the best decoded solution satisfies constraints.
        runtime_seconds: Wall time for the solve call.
        metadata: Extra fields such as ``initial_energy``, ``energy_history``,
            ``best_sequence``, ``best_bitstring``, ``best_binary``,
            ``real_cost``, sample histograms, etc.

    """

    solver_name: str
    objective_value: float
    feasible: bool
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SolverProtocol(Protocol):
    """Contract implemented by every solver backend."""

    solver_name: str

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        """Run the solver on the given instance.

        Args:
            instance: Problem instance (precedences, hotel and travel prices).
            run_config: Backend-specific limits, formulation, QAOA/SA options.

        Returns:
            ``SolverResult`` with best objective, feasibility flag, runtime, and
            optional ``metadata``.

        """
