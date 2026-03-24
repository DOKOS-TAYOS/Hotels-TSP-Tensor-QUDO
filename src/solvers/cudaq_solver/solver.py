"""CUDA-Q backend for QAOA (TQUDO virtual and QUBO formulations)."""

from __future__ import annotations

import math
from typing import Any, Callable

from instance_gen_process import ProblemInstance
from solvers._qaoa_base import BaseQAOASolver
from utils.qaoa_helpers import measurement_histogram_for_json


class CudaqSolver(BaseQAOASolver):
    """CUDA-Q solver using QAOA for TQUDO virtual or QUBO formulations."""

    solver_name = "cudaq"

    def _get_tqudo_runner(self) -> Callable[..., dict] | None:
        """Native TQUDO is not supported on CUDA-Q."""
        return None

    def _get_tqudo_virtual_runner(self) -> Callable[..., dict]:
        """Return emulated TQUDO :func:`run_qaoa` for CUDA-Q."""
        from solvers.cudaq_solver.qaoa_circuit_tqudo import run_qaoa
        return run_qaoa

    def _get_qubo_runner(self) -> Callable[..., dict]:
        """Return QUBO :func:`run_qaoa` for CUDA-Q."""
        from solvers.cudaq_solver.qaoa_circuit_qubo import run_qaoa
        return run_qaoa

    def _serialize_samples(self, samples: Any) -> dict[str, int] | None:
        """Delegate to :func:`~utils.qaoa_helpers.measurement_histogram_for_json`."""
        return measurement_histogram_for_json(samples)

    def _noise_qubit_count(
        self, instance: ProblemInstance, formulation: str,
    ) -> tuple[int, dict[str, Any]]:
        """Return qubit count for noise warnings (virtual TQUDO vs QUBO)."""
        n_available = instance.n_cities - 1
        if formulation == "tqudo_virtual":
            n_qubits = (n_available - 1) * math.ceil(math.log2(n_available))
            return n_qubits, {}
        return n_available * n_available, {}
