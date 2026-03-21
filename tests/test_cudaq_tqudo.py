"""Regression tests for the CUDA-Q Tensor-QUDO circuit."""

from __future__ import annotations

import numpy as np
import pytest

from instance_gen_process import InstanceConfig, generate_TQUDO_from_problem, generate_random_instance
from instance_gen_process.models import RestrictionConfig


def _synthetic_tqudo_tensors(
    n_qudits: int,
    dimension_qudits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build small sparse tensors that exercise both local and long-range phases."""
    Etab = np.zeros((n_qudits, dimension_qudits, dimension_qudits), dtype=float)
    Ettprimeab = np.zeros(
        (n_qudits, n_qudits, dimension_qudits, dimension_qudits),
        dtype=float,
    )
    Etab[0, 0, dimension_qudits - 1] = 1.0
    if n_qudits > 2:
        Etab[1, min(1, dimension_qudits - 1), max(0, dimension_qudits - 2)] = 0.5
        Ettprimeab[0, n_qudits - 1, dimension_qudits - 1, 0] = 0.75
    return Etab, Ettprimeab


@pytest.mark.parametrize(
    ("dimension_qudits", "qubits_per_qudit"),
    [(2, 1), (4, 2), (8, 3)],
)
def test_cudaq_tqudo_builder_kernel_supports_multiple_qudit_widths(
    dimension_qudits: int,
    qubits_per_qudit: int,
) -> None:
    """The CUDA-Q Tensor-QUDO kernel builder should scale without width-specific branches."""
    cudaq = pytest.importorskip("cudaq")
    from solvers.cudaq_solver.qaoa_circuit_tqudo import create_qaoa_ansatz

    Etab, Ettprimeab = _synthetic_tqudo_tensors(n_qudits=3, dimension_qudits=dimension_qudits)

    cudaq.set_target("qpp-cpu")
    try:
        kernel = create_qaoa_ansatz(depth=1, Etab=Etab, Ettprimeab=Ettprimeab)
        samples = cudaq.sample(kernel, [0.2], [0.1], shots_count=8)
    finally:
        cudaq.reset_target()

    assert samples.get_total_shots() == 8
    assert all(len(bitstring) == 3 * qubits_per_qudit for bitstring, _ in samples.items())


def test_cudaq_tqudo_qaoa_runs_on_cpu_simulator_when_target_is_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TQUDO CUDA-Q circuit should compile and execute on a simulator for regression coverage."""
    cudaq = pytest.importorskip("cudaq")
    from solvers.cudaq_solver import qaoa_circuit_tqudo

    config = InstanceConfig(
        n_cities=5,
        n_precedences_range=(0, 0),
        prices_range_hotels=(1.0, 2.0),
        prices_range_travels=(1.0, 2.0),
        seed=17,
    )
    instance = generate_random_instance(config, config.seed)
    # Keep the simulator override local to this test so solver runtime still requires GPU.
    monkeypatch.setattr(
        qaoa_circuit_tqudo,
        "ensure_cudaq_target",
        lambda noise_config=None: cudaq.set_target("qpp-cpu"),
    )

    problem = generate_TQUDO_from_problem(
        instance,
        RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0),
    )
    raw = qaoa_circuit_tqudo.run_qaoa(
        problem.Etab,
        problem.Ettprimeab,
        depth=1,
        max_iter=4,
        n_shots=16,
        sample_shots=16,
        seed=17,
    )

    assert isinstance(raw["energy"], float)
    assert isinstance(raw["best_bitstring"], str)
    assert raw["best_sequence"].shape == (instance.n_cities - 1,)
