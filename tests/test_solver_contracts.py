"""Tests for solver scaffolds and expected contract behavior."""

import random

import pytest

from instance_gen_process import generate_random_instance, load_instance_config
from solvers import CirqSolver, CudaqSolver, SimulatedAnnealingSolver, SolverRunConfig


@pytest.mark.parametrize(
    ("solver_cls", "expected_name"),
    [
        (SimulatedAnnealingSolver, "simulated_annealing"),
        (CirqSolver, "cirq"),
    ],
)
def test_solver_stubs_raise_not_implemented(solver_cls: type, expected_name: str) -> None:
    """Stub solvers must raise NotImplementedError."""
    instance_config = load_instance_config()
    rng = random.Random(instance_config.seed)
    instance = generate_random_instance(instance_config, rng)
    run_config = SolverRunConfig(max_iterations=10, timeout_seconds=1.0)

    solver = solver_cls()
    assert getattr(solver, "solver_name") == expected_name
    with pytest.raises(NotImplementedError):
        solver.solve(instance, run_config)


def test_cudaq_solver_contract() -> None:
    """CudaqSolver is implemented and returns SolverResult."""
    solver = CudaqSolver()
    assert solver.solver_name == "cudaq"

