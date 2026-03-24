"""Backward-compatible exports for :mod:`experiments.parallel_solve_batch`.

The implementation lives in ``parallel_solve_batch``; CUDA-Q is only one of the
backends (Cirq, brute_force, simulated_annealing, cudaq). Tests may monkeypatch
:class:`~concurrent.futures.ProcessPoolExecutor` on this module.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ProcessPoolExecutor, wait

from experiments.parallel_solve_batch import (
    CPU_PARALLEL_ENV,
    CUDAQ_PARALLEL_ENV,
    EXPERIMENT_CUDA_WORKER_ENV,
    ParallelSolveBatchResult as CudaqParallelBatchResult,
    ParallelSolveJob,
    _compact_parallel_status_line,
    _parallel_solve_one_worker,
    _payload_from_future_failure,
    resolve_cpu_max_parallel_instances,
    resolve_cudaq_max_parallel_instances,
    run_parallel_solve_batch as run_cudaq_parallel_batch,
)

CudaqParallelJobSpec = ParallelSolveJob
CudaqParallelJob = ParallelSolveJob

__all__ = [
    "CPU_PARALLEL_ENV",
    "CUDAQ_PARALLEL_ENV",
    "CudaqParallelBatchResult",
    "CudaqParallelJob",
    "CudaqParallelJobSpec",
    "EXPERIMENT_CUDA_WORKER_ENV",
    "Future",
    "ProcessPoolExecutor",
    "CancelledError",
    "FIRST_COMPLETED",
    "wait",
    "_compact_parallel_status_line",
    "_parallel_solve_one_worker",
    "_payload_from_future_failure",
    "resolve_cpu_max_parallel_instances",
    "resolve_cudaq_max_parallel_instances",
    "run_cudaq_parallel_batch",
]
