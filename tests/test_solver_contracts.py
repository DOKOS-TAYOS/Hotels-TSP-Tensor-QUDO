"""Tests for solver scaffolds and expected contract behavior."""

import pytest

from instance_gen_process import generate_random_instance, load_instance_config
from solvers import CirqSolver, CudaqSolver, SimulatedAnnealingSolver, SolverRunConfig


@pytest.mark.parametrize(
    ("solver_cls", "expected_name"),
    [
        (SimulatedAnnealingSolver, "simulated_annealing"),
        (CirqSolver, "cirq"),
        (CudaqSolver, "cudaq"),
    ],
)
def test_solver_stubs_raise_not_implemented(solver_cls: type, expected_name: str) -> None:
    instance_config = load_instance_config()
    instance = generate_random_instance(instance_config)
    run_config = SolverRunConfig(max_iterations=10, timeout_seconds=1.0)

    solver = solver_cls()
    assert getattr(solver, "solver_name") == expected_name
    with pytest.raises(NotImplementedError):
        solver.solve(instance, run_config)

