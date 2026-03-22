"""CUDA-Q backend for QAOA (TQUDO virtual and QUBO formulations)."""

from __future__ import annotations

import math
from typing import Any, Callable

from instance_gen_process import ProblemInstance
from solvers._qaoa_base import BaseQAOASolver


def _serialize_samples(samples: Any) -> dict[str, int] | None:
    """Convert CUDA-Q sample results to a sorted histogram dict.

    Args:
        samples: CUDA-Q sample object (mapping-like) or None.

    Returns:
        ``{bitstring: count}`` sorted by descending count, or None.
    """
    if samples is None:
        return None
    counts: dict[str, int] = {bs: int(cnt) for bs, cnt in samples.items()}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


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
        """Delegate to module-level :func:`_serialize_samples`."""
        return _serialize_samples(samples)

    def _noise_qubit_count(
        self, instance: ProblemInstance, formulation: str,
    ) -> tuple[int, dict[str, Any]]:
        """Return qubit count for noise warnings (virtual TQUDO vs QUBO)."""
        n_available = instance.n_cities - 1
        if formulation == "tqudo_virtual":
            n_qubits = (n_available - 1) * math.ceil(math.log2(n_available))
            return n_qubits, {}
        return n_available * n_available, {}
