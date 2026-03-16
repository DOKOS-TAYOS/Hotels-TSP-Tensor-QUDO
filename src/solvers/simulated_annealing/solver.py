"""Simulated annealing backend for QUBO and TQUDO formulations."""

from __future__ import annotations

import time

import numpy as np

from instance_gen_process import (
    ProblemInstance,
    generate_QUBO_from_problem,
    generate_TQUDO_from_problem,
)
from instance_gen_process.models import ProblemQUBO, ProblemTQUDO, RestrictionConfig
from solvers.base import SolverResult, SolverRunConfig
from utils.constraints import (
    sequence_to_qubo_binary,
    validate_solution_constraints_qubo,
    validate_solution_constraints_tqudo,
)
from utils.costs import calculate_qubo_cost, calculate_real_cost, calculate_tqudo_cost


def _default_restriction() -> RestrictionConfig:
    """Default penalty coefficients for constraint encoding."""
    return RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0)


def _evaluate_cost(
    formulation: str,
    problem: ProblemQUBO | ProblemTQUDO,
    sequence: np.ndarray,
    n_available: int,
) -> float:
    """Evaluate cost for a sequence given the formulation."""
    if formulation == "tqudo":
        return float(calculate_tqudo_cost(problem, sequence))
    # QUBO: convert sequence to binary first
    binary = sequence_to_qubo_binary(sequence, n_available)
    return float(calculate_qubo_cost(problem, binary))


def _swap_neighbor(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of sequence with two random positions swapped."""
    neighbor = sequence.copy()
    i, j = rng.integers(0, len(sequence), size=2)
    while i == j:
        j = rng.integers(0, len(sequence))
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


class SimulatedAnnealingSolver:
    """Simulated annealing solver for QUBO and TQUDO formulations."""

    solver_name = "simulated_annealing"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        """Run simulated annealing and return standardized result."""
        restriction = run_config.restriction_config or _default_restriction()
        formulation = run_config.formulation
        n_available = instance.n_cities - 1

        rng = np.random.default_rng(run_config.seed)

        if formulation == "tqudo":
            problem = generate_TQUDO_from_problem(instance, restriction)
        else:
            problem = generate_QUBO_from_problem(instance, restriction)

        start = time.perf_counter()

        # Initial random permutation
        current = rng.permutation(n_available).astype(np.int64)
        current_cost = _evaluate_cost(formulation, problem, current, n_available)
        initial_energy = current_cost
        best = current.copy()
        best_cost = current_cost
        energy_history: list[float] = [initial_energy]

        # SA parameters
        T_initial = 1000.0
        T_final = 1e-6
        alpha = 0.995
        T = T_initial

        max_iter = run_config.max_iterations
        timeout = run_config.timeout_seconds

        for iteration in range(max_iter):
            if timeout is not None and (time.perf_counter() - start) >= timeout:
                break

            neighbor = _swap_neighbor(current, rng)
            neighbor_cost = _evaluate_cost(formulation, problem, neighbor, n_available)

            delta = neighbor_cost - current_cost
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost

            energy_history.append(current_cost)
            T = max(T_final, T * alpha)

        runtime_seconds = time.perf_counter() - start

        # Validate and build result
        best_sequence = best.tolist()
        if formulation == "tqudo":
            feasible = validate_solution_constraints_tqudo(instance, best_sequence)
        else:
            best_binary = sequence_to_qubo_binary(best, n_available)
            feasible = validate_solution_constraints_qubo(instance, best_binary)

        metadata: dict[str, object] = {
            "best_sequence": best_sequence,
            "initial_energy": initial_energy,
            "energy_history": energy_history,
        }
        if feasible:
            metadata["real_cost"] = float(
                calculate_real_cost(instance, best_sequence)
            )

        return SolverResult(
            solver_name=self.solver_name,
            objective_value=best_cost,
            feasible=feasible,
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )
