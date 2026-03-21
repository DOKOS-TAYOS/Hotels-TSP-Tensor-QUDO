"""Tests for solver scaffolds and expected contract behavior."""

import warnings

import pytest
from numpy.exceptions import ComplexWarning

from instance_gen_process import InstanceConfig, generate_random_instance
from solvers import CirqSolver, CudaqSolver, SimulatedAnnealingSolver, SolverRunConfig


def _contract_test_config() -> InstanceConfig:
    """Use a small fixed instance size so solver contract tests stay safe."""
    return InstanceConfig(
        n_cities=5,
        n_precedences_range=(2, 3),
        prices_range_hotels=(30.0, 150.0),
        prices_range_travels=(30.0, 150.0),
        seed=42,
    )


def _cudaq_qubo_test_config() -> InstanceConfig:
    """Use a compact QUBO-friendly instance for CUDA-Q smoke tests."""
    return InstanceConfig(
        n_cities=4,
        n_precedences_range=(0, 0),
        prices_range_hotels=(30.0, 150.0),
        prices_range_travels=(30.0, 150.0),
        seed=7,
    )


@pytest.mark.parametrize("formulation", ["qubo", "tqudo"])
def test_cudaq_solver_requires_nvidia_gpu(
    formulation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CudaqSolver must not fall back silently to CPU when no GPU is available."""
    pytest.importorskip("cudaq")
    from solvers.cudaq_solver import cudaq_target

    instance_config = _cudaq_qubo_test_config() if formulation == "qubo" else _contract_test_config()
    instance = generate_random_instance(instance_config, instance_config.seed)
    run_config = SolverRunConfig(
        formulation=formulation,
        qaoa_depth=1,
        qaoa_max_iter=4,
        qaoa_shots=32,
        qaoa_sample_shots=32,
        seed=42,
    )

    monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 0)
    monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: True)

    def _unexpected_set_target(*args, **kwargs):
        raise AssertionError("set_target should not be called when no GPU is available")

    monkeypatch.setattr(cudaq_target.cudaq, "set_target", _unexpected_set_target)

    solver = CudaqSolver()
    with pytest.raises(RuntimeError, match="requires an NVIDIA GPU"):
        solver.solve(instance, run_config)


def test_cudaq_solver_contract_qubo_on_nvidia_gpu() -> None:
    """CudaqSolver should return SolverResult when a real NVIDIA target is available."""
    cudaq = pytest.importorskip("cudaq")
    if cudaq.num_available_gpus() < 1 or not cudaq.has_target("nvidia"):
        pytest.skip("CUDA-Q NVIDIA target unavailable in this environment")

    instance_config = _cudaq_qubo_test_config()
    instance = generate_random_instance(instance_config, instance_config.seed)
    run_config = SolverRunConfig(
        formulation="qubo",
        qaoa_depth=1,
        qaoa_max_iter=4,
        qaoa_shots=32,
        qaoa_sample_shots=32,
        seed=42,
    )

    solver = CudaqSolver()
    result = solver.solve(instance, run_config)

    assert result.solver_name == "cudaq"
    assert isinstance(result.objective_value, (int, float))
    assert isinstance(result.feasible, bool)
    assert isinstance(result.runtime_seconds, (int, float))
    assert result.runtime_seconds >= 0
    assert "best_bitstring" in result.metadata
    assert "best_binary" in result.metadata


def test_cudaq_solver_contract_tqudo_on_nvidia_gpu() -> None:
    """CudaqSolver TQUDO should return SolverResult when a real NVIDIA target is available."""
    cudaq = pytest.importorskip("cudaq")
    if cudaq.num_available_gpus() < 1 or not cudaq.has_target("nvidia"):
        pytest.skip("CUDA-Q NVIDIA target unavailable in this environment")

    instance_config = _contract_test_config()
    instance = generate_random_instance(instance_config, instance_config.seed)
    run_config = SolverRunConfig(
        formulation="tqudo",
        qaoa_depth=1,
        qaoa_max_iter=4,
        qaoa_shots=32,
        qaoa_sample_shots=32,
        seed=42,
    )

    solver = CudaqSolver()
    result = solver.solve(instance, run_config)

    assert result.solver_name == "cudaq"
    assert isinstance(result.objective_value, (int, float))
    assert isinstance(result.feasible, bool)
    assert isinstance(result.runtime_seconds, (int, float))
    assert result.runtime_seconds >= 0
    assert "best_sequence" in result.metadata
    assert "best_bitstring" in result.metadata


@pytest.mark.parametrize("formulation", ["tqudo", "qubo"])
def test_cirq_solver_contract(formulation: str) -> None:
    """CirqSolver is implemented and returns SolverResult."""
    pytest.importorskip("cirq")
    instance_config = _contract_test_config()
    instance = generate_random_instance(instance_config, instance_config.seed)
    run_config = SolverRunConfig(
        max_iterations=10,
        formulation=formulation,
        qaoa_depth=1,
        qaoa_max_iter=4,
        qaoa_shots=20,
        qaoa_sample_shots=50,
        seed=42,
    )

    solver = CirqSolver()
    assert solver.solver_name == "cirq"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
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
    instance_config = _contract_test_config()
    instance = generate_random_instance(instance_config, instance_config.seed)
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
    instance_config = _contract_test_config()
    instance = generate_random_instance(instance_config, instance_config.seed)
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
