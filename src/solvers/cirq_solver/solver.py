"""Cirq backend for QAOA (TQUDO, TQUDO virtual, and QUBO formulations)."""

from __future__ import annotations

from typing import Any, Callable

from instance_gen_process import ProblemInstance
from solvers._qaoa_base import BaseQAOASolver
from utils.qaoa_helpers import measurement_histogram_for_json


class CirqSolver(BaseQAOASolver):
    """Cirq solver using QAOA for TQUDO, TQUDO virtual, or QUBO formulations."""

    solver_name = "cirq"

    def _get_tqudo_runner(self) -> Callable[..., dict]:
        """Return native-qudit TQUDO :func:`run_qaoa`."""
        from solvers.cirq_solver.qaoa_circuit_tqudo import run_qaoa
        return run_qaoa

    def _get_tqudo_virtual_runner(self) -> Callable[..., dict]:
        """Return qubit-emulated TQUDO :func:`run_qaoa`."""
        from solvers.cirq_solver.qaoa_circuit_tqudo_qubit_emulation import run_qaoa
        return run_qaoa

    def _get_qubo_runner(self) -> Callable[..., dict]:
        """Return QUBO :func:`run_qaoa`."""
        from solvers.cirq_solver.qaoa_circuit_qubo import run_qaoa
        return run_qaoa

    def _serialize_samples(self, samples: Any) -> dict[str, int] | None:
        """See :func:`~utils.qaoa_helpers.measurement_histogram_for_json`."""
        return measurement_histogram_for_json(samples)

    def _noise_qubit_count(
        self, instance: ProblemInstance, formulation: str,
    ) -> tuple[int, dict[str, Any]]:
        """Return qudit/qubit counts for noise warnings (native vs virtual vs QUBO)."""
        n_available = instance.n_cities - 1
        if formulation == "tqudo":
            return n_available - 1, {"qudit_dimension": n_available}
        if formulation == "tqudo_virtual":
            import math
            n_qubits = (n_available - 1) * math.ceil(math.log2(n_available))
            return n_qubits, {}
        return n_available * n_available, {}
