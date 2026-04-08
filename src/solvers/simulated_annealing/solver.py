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
from utils.costs import (
    calculate_qubo_cost_from_sequence,
    calculate_real_cost,
    calculate_tqudo_cost,
)
from utils.progress import reporter


def _default_restriction() -> RestrictionConfig:
    """Return default QUBO/TQUDO penalty weights when none are supplied."""
    return RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0)


def evaluate_cost(
    formulation: str,
    problem: ProblemQUBO | ProblemTQUDO,
    sequence: np.ndarray,
    n_available: int,
) -> float:
    """Return objective value for a permutation under the active formulation.

    Args:
        formulation: ``tqudo`` or ``qubo``.
        problem: Built QUBO or TQUDO problem.
        sequence: City indices per timestep.
        n_available: Number of cities excluding the depot.

    Returns:
        Cost in stored (normalized) units, rescaled consistently with solvers.

    """
    if formulation == "tqudo":
        return float(calculate_tqudo_cost(problem, sequence))
    assert isinstance(problem, ProblemQUBO)
    return calculate_qubo_cost_from_sequence(problem, sequence, n_available)


def _tqudo_swap_delta(
    problem: ProblemTQUDO,
    x: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Return (cost_after_swap - cost_before) for swapping ``x[i]`` and ``x[j]``.

    Same algebra as :func:`~utils.costs.calculate_tqudo_cost` but only touches
    Etab edges and Ettprimeab pairs that involve timesteps ``i`` or ``j`` (O(n)
    terms vs O(n^2) for a full cost).
    """
    if i == j:
        return 0.0
    if i > j:
        i, j = j, i
    Etab = problem.Etab
    Ett = problem.Ettprimeab
    es = problem.energy_scale
    n = int(x.shape[0])
    x_new = x.copy()
    x_new[i], x_new[j] = x_new[j], x_new[i]

    delta_etab = 0.0
    affected_t: set[int] = set()
    for t in (i - 1, i, j - 1, j):
        if 0 <= t < n - 1:
            affected_t.add(t)
    for t in affected_t:
        delta_etab += float(Etab[t, x_new[t], x_new[t + 1]] - Etab[t, x[t], x[t + 1]])

    delta_ett = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            if a not in (i, j) and b not in (i, j):
                continue
            delta_ett += float(Ett[a, b, x_new[a], x_new[b]] - Ett[a, b, x[a], x[b]])

    return (delta_etab + delta_ett) * es


def _swap_neighbor(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a neighbour by swapping two uniformly random positions.

    Args:
        sequence: Current permutation.
        rng: NumPy random generator.

    Returns:
        New permutation (copy); unchanged if length < 2.

    """
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
    """Return a neighbour by moving one city to a different index.

    Args:
        sequence: Current permutation.
        rng: NumPy random generator.

    Returns:
        New permutation; copy only if length < 2.

    """
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
    """Return a neighbour by reversing a random contiguous segment (2-opt style).

    Args:
        sequence: Current permutation.
        rng: NumPy random generator.

    Returns:
        New permutation; falls back to :func:`_swap_neighbor` if length < 3.

    """
    n = len(sequence)
    if n < 3:
        return _swap_neighbor(sequence, rng)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    neighbor = sequence.copy()
    neighbor[i : j + 1] = neighbor[i : j + 1][::-1]
    return neighbor


def random_neighbor(
    sequence: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Apply swap, insert, or reverse with uniform probability.

    Args:
        sequence: Current permutation.
        rng: NumPy random generator.

    Returns:
        Neighbour state and ``op_id`` in ``{0: swap, 1: insert, 2: reverse}``.

    """
    _operators = (_swap_neighbor, _insert_neighbor, _reverse_neighbor)
    op_id = int(rng.integers(0, len(_operators)))
    return _operators[op_id](sequence, rng), op_id


class SimulatedAnnealingSolver:
    """Simulated annealing solver for QUBO and TQUDO formulations."""

    solver_name = "simulated_annealing"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        """Minimize QUBO or TQUDO cost by simulated annealing over permutations.

        Args:
            instance: Problem instance.
            run_config: Temperature schedule, iteration cap, formulation, seed.

        Returns:
            :class:`~solvers.base.SolverResult` with best cost and metadata.

        Raises:
            ValueError: If ``formulation`` is ``tqudo_virtual``.

        """
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
        current_cost = evaluate_cost(formulation, problem, current, n_available)
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

            neighbor, op_id = random_neighbor(current, rng)
            if formulation == "tqudo" and op_id == 0:
                assert isinstance(problem, ProblemTQUDO)
                diff = np.flatnonzero(current != neighbor)
                if diff.size == 2:
                    i0, j0 = int(diff[0]), int(diff[1])
                    neighbor_cost = current_cost + _tqudo_swap_delta(
                        problem,
                        current,
                        i0,
                        j0,
                    )
                else:
                    neighbor_cost = evaluate_cost(
                        formulation,
                        problem,
                        neighbor,
                        n_available,
                    )
            else:
                neighbor_cost = evaluate_cost(
                    formulation,
                    problem,
                    neighbor,
                    n_available,
                )

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
            metadata["real_cost"] = float(calculate_real_cost(instance, best_sequence))

        return SolverResult(
            solver_name=self.solver_name,
            objective_value=best_cost,
            feasible=feasible,
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )
