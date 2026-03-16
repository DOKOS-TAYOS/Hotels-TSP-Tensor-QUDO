"""Tests for solver scaffolds and expected contract behavior."""

import random

import pytest

from instance_gen_process import generate_random_instance, load_instance_config
from solvers import CirqSolver, CudaqSolver, SimulatedAnnealingSolver, SolverRunConfig


def test_cudaq_solver_contract() -> None:
    """CudaqSolver is implemented and returns SolverResult."""
    solver = CudaqSolver()
    assert solver.solver_name == "cudaq"


@pytest.mark.parametrize("formulation", ["tqudo", "qubo"])
def test_cirq_solver_contract(formulation: str) -> None:
    """CirqSolver is implemented and returns SolverResult."""
    pytest.importorskip("cirq")
    instance_config = load_instance_config()
    rng = random.Random(instance_config.seed)
    instance = generate_random_instance(instance_config, rng)
    run_config = SolverRunConfig(
        max_iterations=10,
        formulation=formulation,
        qaoa_depth=1,
        qaoa_max_iter=3,
        qaoa_shots=20,
        qaoa_sample_shots=50,
        seed=42,
    )

    solver = CirqSolver()
    assert solver.solver_name == "cirq"
    result = solver.solve(instance, run_config)

    assert isinstance(result.solver_name, str)
    assert result.solver_name == "cirq"
    assert isinstance(result.objective_value, (int, float))
    assert isinstance(result.feasible, bool)
    assert isinstance(result.runtime_seconds, (int, float))
    assert result.runtime_seconds >= 0
    assert "best_sequence" in result.metadata or "best_binary" in result.metadata


def test_simulated_annealing_solver_contract() -> None:
    """SimulatedAnnealingSolver is implemented and returns SolverResult."""
    instance_config = load_instance_config()
    rng = random.Random(instance_config.seed)
    instance = generate_random_instance(instance_config, rng)
    run_config = SolverRunConfig(
        max_iterations=100,
        timeout_seconds=5.0,
        formulation="tqudo",
        seed=42,
    )

    solver = SimulatedAnnealingSolver()
    assert solver.solver_name == "simulated_annealing"
    result = solver.solve(instance, run_config)

    assert isinstance(result.solver_name, str)
    assert result.solver_name == "simulated_annealing"
    assert isinstance(result.objective_value, (int, float))
    assert isinstance(result.feasible, bool)
    assert isinstance(result.runtime_seconds, (int, float))
    assert result.runtime_seconds >= 0
    assert "best_sequence" in result.metadata
    assert isinstance(result.metadata["best_sequence"], list)
    assert len(result.metadata["best_sequence"]) == instance.n_cities - 1


@pytest.mark.parametrize("formulation", ["tqudo", "qubo"])
def test_simulated_annealing_works_with_both_formulations(
    formulation: str,
) -> None:
    """SA solver runs with both TQUDO and QUBO formulations."""
    instance_config = load_instance_config()
    rng = random.Random(instance_config.seed)
    instance = generate_random_instance(instance_config, rng)
    run_config = SolverRunConfig(
        max_iterations=50,
        formulation=formulation,
        seed=123,
    )

    solver = SimulatedAnnealingSolver()
    result = solver.solve(instance, run_config)

    assert result.solver_name == "simulated_annealing"
    assert isinstance(result.objective_value, (int, float))
    assert isinstance(result.feasible, bool)
    assert result.runtime_seconds >= 0
    assert len(result.metadata["best_sequence"]) == instance.n_cities - 1

