"""CUDA-Q backend for QAOA (TQUDO virtual and QUBO formulations)."""

from __future__ import annotations

import math
from typing import Any, Callable

from instance_gen_process import ProblemInstance
from solvers._qaoa_base import BaseQAOASolver


def _serialize_samples(samples: Any) -> dict[str, int] | None:
    """Convert a cudaq.SampleResult to a JSON-serializable dict.

    Returns a mapping of bitstring -> count, or None if samples is None.
    Sorted by count descending for readability.
    """
    if samples is None:
        return None
    counts: dict[str, int] = {bs: int(cnt) for bs, cnt in samples.items()}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


class CudaqSolver(BaseQAOASolver):
    """CUDA-Q solver using QAOA for TQUDO virtual or QUBO formulations."""

    solver_name = "cudaq"

    def _get_tqudo_runner(self) -> Callable[..., dict] | None:
        return None

    def _get_tqudo_virtual_runner(self) -> Callable[..., dict]:
        from solvers.cudaq_solver.qaoa_circuit_tqudo import run_qaoa
        return run_qaoa

    def _get_qubo_runner(self) -> Callable[..., dict]:
        from solvers.cudaq_solver.qaoa_circuit_qubo import run_qaoa
        return run_qaoa

    def _serialize_samples(self, samples: Any) -> dict[str, int] | None:
        return _serialize_samples(samples)

    def _noise_qubit_count(
        self, instance: ProblemInstance, formulation: str,
    ) -> tuple[int, dict[str, Any]]:
        n_available = instance.n_cities - 1
        if formulation == "tqudo_virtual":
            n_qubits = (n_available - 1) * math.ceil(math.log2(n_available))
            return n_qubits, {}
        return n_available * n_available, {}
