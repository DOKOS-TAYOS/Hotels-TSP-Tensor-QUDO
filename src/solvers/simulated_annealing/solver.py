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
from utils.progress import reporter


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
    """Return a copy of *sequence* with two random positions swapped."""
    n = len(sequence)
    if n < 2:
        return sequence.copy()
    neighbor = sequence.copy()
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1
    tmp = neighbor[i].copy()
    neighbor[i] = neighbor[j]
    neighbor[j] = tmp
    return neighbor


def _insert_neighbor(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remove a random element and re-insert it at a different position."""
    n = len(sequence)
    if n < 2:
        return sequence.copy()
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1  # skip position i → j ∈ {0..n-1} \ {i}
    element = sequence[i]
    neighbor = np.delete(sequence, i)
    # j is a valid np.insert index in the shortened array (length n-1);
    # inserting at any position other than i guarantees a different permutation.
    neighbor = np.insert(neighbor, j, element)
    return neighbor


def _reverse_neighbor(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reverse a random sub-segment (2-opt style move)."""
    n = len(sequence)
    if n < 3:
        return _swap_neighbor(sequence, rng)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    neighbor = sequence.copy()
    neighbor[i : j + 1] = neighbor[i : j + 1][::-1]
    return neighbor


def _random_neighbor(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick a neighborhood operator at random (swap, insert, or reverse)."""
    operators = (_swap_neighbor, _insert_neighbor, _reverse_neighbor)
    op = operators[int(rng.integers(0, len(operators)))]
    return op(sequence, rng)


class SimulatedAnnealingSolver:
    """Simulated annealing solver for QUBO and TQUDO formulations."""

    solver_name = "simulated_annealing"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        """Run simulated annealing and return standardized result."""
        restriction = run_config.restriction_config or _default_restriction()
        formulation = run_config.formulation
        if formulation == "tqudo_virtual":
            raise ValueError(
                "Formulation 'tqudo_virtual' (qubit emulation) is not supported by "
                "simulated annealing. Use 'tqudo' or 'qubo' instead."
            )
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

        # SA parameters (configurable via SolverRunConfig)
        T_initial = run_config.sa_t_initial
        T_final = run_config.sa_t_final
        alpha = run_config.sa_alpha
        T = T_initial

        max_iter = run_config.max_iterations
        timeout = run_config.timeout_seconds
        iterations_completed = 0

        for iteration in range(max_iter):
            if timeout is not None and (time.perf_counter() - start) >= timeout:
                break
            if T <= T_final:
                break

            neighbor = _random_neighbor(current, rng)
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
            iterations_completed = iteration + 1
            reporter.opt_step(iterations_completed, max_iter, current_cost)

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
            "iterations_completed": iterations_completed,
            "final_temperature": T,
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
