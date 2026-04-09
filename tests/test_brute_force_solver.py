"""Tests for exhaustive brute_force solver (QUBO and TQUDO spaces)."""

from __future__ import annotations

import numpy as np
import pytest

from instance_gen_process.models import ProblemInstance, RestrictionConfig
from solvers.base import SolverRunConfig
from solvers.brute_force import BruteForceSolver
from utils.constraints import (
    validate_solution_constraints_qubo,
    validate_solution_constraints_tqudo,
)
from utils.costs import calculate_qubo_cost, calculate_real_cost, calculate_tqudo_cost

from instance_gen_process import generate_QUBO_from_problem, generate_TQUDO_from_problem


def _tiny_instance() -> ProblemInstance:
    """n_cities=3 => two visitable cities; small full enumeration spaces."""
    n = 3
    pt = np.ones((n, n, n), dtype=np.float64) * 5.0
    return ProblemInstance(
        n_cities=n,
        precedences=(),
        prices_hotels=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        prices_travels=pt,
        seed=0,
    )


def _reference_min_tqudo(
    instance: ProblemInstance, restriction: RestrictionConfig
) -> tuple[float, list[int]]:
    problem = generate_TQUDO_from_problem(instance, restriction)
    n_available = instance.n_cities - 1
    best_c = float("inf")
    best_s: list[int] | None = None
    for a in range(n_available):
        for b in range(n_available):
            seq = [a, b]
            c = calculate_tqudo_cost(problem, np.array(seq, dtype=int))
            if c < best_c:
                best_c = c
                best_s = seq
    assert best_s is not None
    return best_c, best_s


def _reference_min_qubo(
    instance: ProblemInstance, restriction: RestrictionConfig
) -> tuple[float, np.ndarray]:
    problem = generate_QUBO_from_problem(instance, restriction)
    n_available = instance.n_cities - 1
    n_vars = n_available * n_available
    best_c = float("inf")
    best_x: np.ndarray | None = None
    x = np.zeros(n_vars, dtype=np.float64)
    for i in range(1 << n_vars):
        v = i
        for b in range(n_vars):
            x[b] = float(v & 1)
            v >>= 1
        c = calculate_qubo_cost(problem, x)
        if c < best_c:
            best_c = c
            best_x = x.copy()
    assert best_x is not None
    return best_c, best_x


def test_brute_force_tqudo_matches_reference() -> None:
    instance = _tiny_instance()
    restriction = RestrictionConfig(lambda_0=10.0, lambda_1=10.0, lambda_2=100.0)
    ref_cost, _ = _reference_min_tqudo(instance, restriction)
    run = SolverRunConfig(
        formulation="tqudo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )
    out = BruteForceSolver().solve(instance, run)
    assert out.objective_value == pytest.approx(ref_cost)
    assert out.metadata["configs_evaluated"] == 2**2
    bfs = out.metadata["best_feasible_sequence"]
    assert bfs is not None
    assert validate_solution_constraints_tqudo(instance, bfs)


def test_brute_force_qubo_matches_reference() -> None:
    instance = _tiny_instance()
    restriction = RestrictionConfig(lambda_0=10.0, lambda_1=10.0, lambda_2=100.0)
    ref_cost, ref_x = _reference_min_qubo(instance, restriction)
    run = SolverRunConfig(
        formulation="qubo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )
    out = BruteForceSolver().solve(instance, run)
    assert out.objective_value == pytest.approx(ref_cost)
    assert out.metadata["configs_evaluated"] == 2**4
    assert np.allclose(np.array(out.metadata["best_binary"]), ref_x)
    assert out.metadata["best_feasible_sequence"] is not None


def test_brute_force_global_minimum_can_be_infeasible_with_zero_penalties() -> None:
    """When penalties are off, a non-tour sequence can beat every permutation.

    Exercises objective_value vs best_feasible_* metadata (solver must not conflate them).
    """
    n = 3
    n_available = 2
    ph = np.zeros((n_available, n_available), dtype=np.float64)
    pt = np.zeros((n, n, n), dtype=np.float64)
    # Strongly favour staying on city 0 across the middle leg (duplicate visit).
    pt[1, 0, 0] = -1000.0

    instance = ProblemInstance(
        n_cities=n,
        precedences=(),
        prices_hotels=ph,
        prices_travels=pt,
        seed=0,
    )
    restriction = RestrictionConfig(lambda_0=0.0, lambda_1=0.0, lambda_2=0.0)
    run_t = SolverRunConfig(
        formulation="tqudo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )
    run_q = SolverRunConfig(
        formulation="qubo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )

    tqudo_prob = generate_TQUDO_from_problem(instance, restriction)
    best_t, best_feasible_t = float("inf"), float("inf")
    for a in range(n_available):
        for b in range(n_available):
            seq = [a, b]
            c = calculate_tqudo_cost(tqudo_prob, np.array(seq, dtype=int))
            best_t = min(best_t, c)
            if validate_solution_constraints_tqudo(instance, seq):
                best_feasible_t = min(best_feasible_t, c)

    qubo_prob = generate_QUBO_from_problem(instance, restriction)
    n_vars = n_available * n_available
    best_q = float("inf")
    best_feasible_q = float("inf")
    x = np.zeros(n_vars, dtype=np.float64)
    for i in range(1 << n_vars):
        v = i
        for b in range(n_vars):
            x[b] = float(v & 1)
            v >>= 1
        c = calculate_qubo_cost(qubo_prob, x)
        best_q = min(best_q, c)
        if validate_solution_constraints_qubo(instance, x):
            best_feasible_q = min(best_feasible_q, c)

    assert best_feasible_t > best_t
    assert best_feasible_q > best_q

    out_t = BruteForceSolver().solve(instance, run_t)
    assert out_t.objective_value == pytest.approx(best_t)
    assert out_t.feasible is False
    assert out_t.metadata["best_feasible_sequence"] is not None
    assert out_t.metadata["best_feasible_objective_value"] == pytest.approx(best_feasible_t)

    out_q = BruteForceSolver().solve(instance, run_q)
    assert out_q.objective_value == pytest.approx(best_q)
    assert out_q.feasible is False
    assert out_q.metadata["best_feasible_sequence"] is not None
    assert out_q.metadata["best_feasible_objective_value"] == pytest.approx(best_feasible_q)


def test_brute_force_tqudo_qubo_same_optimal_tour_real_cost() -> None:
    instance = _tiny_instance()
    restriction = RestrictionConfig(lambda_0=10.0, lambda_1=10.0, lambda_2=100.0)
    run_t = SolverRunConfig(
        formulation="tqudo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )
    run_q = SolverRunConfig(
        formulation="qubo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )
    t_res = BruteForceSolver().solve(instance, run_t)
    q_res = BruteForceSolver().solve(instance, run_q)
    t_bf = t_res.metadata["best_feasible_sequence"]
    q_bf = q_res.metadata["best_feasible_sequence"]
    assert t_bf is not None and q_bf is not None
    rc_t = calculate_real_cost(instance, t_bf)
    rc_q = calculate_real_cost(instance, q_bf)
    assert rc_t == pytest.approx(rc_q)


def test_parse_solver_config_includes_brute_caps() -> None:
    from instance_gen_process.solver_config_loader import parse_solver_config_dict

    d = parse_solver_config_dict(
        {
            "n_instances": 1,
            "solver": "brute_force",
            "formulation": "tqudo",
            "brute_force_max_assignments_tqudo": 5000,
            "brute_force_max_assignments_qubo": 9999,
        }
    )
    assert d["brute_force_max_assignments_tqudo"] == 5000
    assert d["brute_force_max_assignments_qubo"] == 9999
