"""Parallel instance solves for on-disk experiment workflows (multi-backend).

Uses ``multiprocessing`` spawn and ``ProcessPoolExecutor`` so each worker owns
its own backend context (CUDA-Q on GPU; Cirq, ``brute_force``, and
``simulated_annealing`` on CPU). The parent shows which instance labels are
currently running.

Note:
    The legacy module name ``cudaq_parallel`` re-exports this API; CUDA-Q is
    only one of the supported backends.
"""

from __future__ import annotations

import gc
import importlib
import logging
import multiprocessing as mp
import os
import re
import shutil
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, wait
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from solvers.base import SolverRunConfig

from utils.experiment_serialize import build_solution_record, serialize_solver_result
from utils.native_stderr import (
    redirect_native_stderr_to_file,
    resolve_native_stderr_log_path,
    silence_native_stderr_requested,
)
from experiments.workflow_io import load_problem_instance_json, serialize_problem_instance

logger = logging.getLogger(__name__)

_INST_LABEL_RE = re.compile(r"\binst=(\d+)\s*$")

# Pool workers set this so ``utils.progress`` stays quiet for parallel experiment batches.
EXPERIMENT_CUDA_WORKER_ENV = "HTSP_EXPERIMENT_CUDA_WORKER"
CUDAQ_PARALLEL_ENV = "HTSP_CUDAQ_MAX_PARALLEL_INSTANCES"
CPU_PARALLEL_ENV = "HTSP_CPU_MAX_PARALLEL_INSTANCES"

# Top-level pickling for spawn: (module, class_name).
_PARALLEL_SOLVER_IMPORTS: dict[str, tuple[str, str]] = {
    "cudaq": ("solvers.cudaq_solver", "CudaqSolver"),
    "cirq": ("solvers.cirq_solver", "CirqSolver"),
    "brute_force": ("solvers.brute_force", "BruteForceSolver"),
    "simulated_annealing": ("solvers.simulated_annealing", "SimulatedAnnealingSolver"),
}


def _compact_parallel_status_line(
    status_prefix: str,
    running_labels: frozenset[str],
    n_finished: int,
    total: int,
    max_columns: int,
) -> str:
    """Build a single-line status that fits *max_columns* (avoids TTY wrap glitches)."""
    ids: list[int] = []
    for lb in sorted(running_labels):
        m = _INST_LABEL_RE.search(lb)
        if m:
            ids.append(int(m.group(1)))
    if ids:
        active = "[" + ",".join(str(i) for i in ids) + "]"
    elif running_labels:
        active = f"{len(running_labels)} workers"
    else:
        active = "[]"
    line = f"{status_prefix} active_inst={active} writes={n_finished}/{total}"
    if len(line) > max_columns and max_columns >= 16:
        return line[: max_columns - 3] + "..."
    return line


def resolve_cudaq_max_parallel_instances(cfg_dict: dict[str, Any]) -> int:
    """Resolve parallel worker count: env overrides YAML ``cudaq_max_parallel_instances``.

    Args:
        cfg_dict: Merged experiment + base solver mapping (may include
            ``cudaq_max_parallel_instances``).

    Returns:
        Integer >= 1.

    """
    raw = os.environ.get(CUDAQ_PARALLEL_ENV)
    if raw is not None and str(raw).strip() != "":
        w = int(raw)
    else:
        w = int(cfg_dict.get("cudaq_max_parallel_instances", 1))
    return max(1, w)


def resolve_cpu_max_parallel_instances(cfg_dict: dict[str, Any]) -> int:
    """Resolve CPU parallel worker count for Cirq, brute_force, and simulated_annealing.

    Env overrides YAML ``cpu_max_parallel_instances``.

    Args:
        cfg_dict: Merged experiment + base solver mapping (may include
            ``cpu_max_parallel_instances``).

    Returns:
        Integer >= 1.

    """
    raw = os.environ.get(CPU_PARALLEL_ENV)
    if raw is not None and str(raw).strip() != "":
        w = int(raw)
    else:
        w = int(cfg_dict.get("cpu_max_parallel_instances", 1))
    return max(1, w)


@dataclass(frozen=True, slots=True)
class ParallelSolveJob:
    """Picklable unit of work for one on-disk parallel batch instance.

    ``status_queue`` is injected by ``run_parallel_solve_batch`` for worker
    processes; omit or leave ``None`` in job specs passed by callers.
    """

    k: int
    instance_json_path: str
    status_label: str
    run_config: SolverRunConfig
    instance_config_dict: dict[str, Any]
    solver_config_serializable: dict[str, Any]
    solver_name: str
    formulation: str
    n_cities: int
    path_depth: int | None
    output_root: str
    status_queue: Any | None = None


def _parallel_solve_one_worker(job: ParallelSolveJob) -> tuple[int, dict[str, Any]]:
    """Run one solver on one instance in a child process (top-level for spawn pickling)."""
    pair = _PARALLEL_SOLVER_IMPORTS.get(job.solver_name)
    if pair is None:
        raise ValueError(f"parallel batch unsupported for solver_name={job.solver_name!r}")
    module_path, class_name = pair
    mod = importlib.import_module(module_path)
    solver_cls = getattr(mod, class_name)

    os.environ[EXPERIMENT_CUDA_WORKER_ENV] = "1"
    q = job.status_queue
    if q is None:
        raise RuntimeError("parallel worker job missing status_queue")
    q.put(("start", job.status_label))
    src = Path(job.instance_json_path)
    _stderr_cm = nullcontext()
    if job.solver_name == "cudaq" and silence_native_stderr_requested():
        _stderr_cm = redirect_native_stderr_to_file(
            resolve_native_stderr_log_path(Path(job.output_root))
        )
    with _stderr_cm:
        try:
            instance = load_problem_instance_json(src)
            result = solver_cls().solve(instance, job.run_config)
            payload = build_solution_record(
                instance=serialize_problem_instance(instance),
                instance_config=job.instance_config_dict,
                instance_index=job.k - 1,
                solver_config=job.solver_config_serializable,
                solver_output=serialize_solver_result(result),
                instance_source=str(src),
            )
            return job.k, payload
        except Exception:
            logger.exception("%s worker failed for %s", job.solver_name, src)
            try:
                instance = load_problem_instance_json(src)
                inst_dict = serialize_problem_instance(instance)
            except Exception:
                inst_dict = {}
            payload = build_solution_record(
                instance=inst_dict,
                instance_config=job.instance_config_dict,
                instance_index=job.k - 1,
                solver_config=job.solver_config_serializable,
                solver_output={
                    "solver_name": job.solver_name,
                    "error": traceback.format_exc(),
                },
                instance_source=str(src),
            )
            return job.k, payload
        finally:
            q.put(("done", job.status_label))


def _payload_from_future_failure(job: ParallelSolveJob, exc_tb: str) -> dict[str, Any]:
    """JSON payload when the worker exits before returning (crash, unpickling, etc.)."""
    inst_dict: dict[str, Any] = {}
    try:
        instance = load_problem_instance_json(Path(job.instance_json_path))
        inst_dict = serialize_problem_instance(instance)
    except Exception:
        pass
    return build_solution_record(
        instance=inst_dict,
        instance_config=job.instance_config_dict,
        instance_index=job.k - 1,
        solver_config=job.solver_config_serializable,
        solver_output={
            "solver_name": job.solver_name,
            "error": exc_tb,
        },
        instance_source=str(job.instance_json_path),
    )


@dataclass(frozen=True, slots=True)
class ParallelSolveBatchResult:
    """Outcome of solving a batch of instances with a process pool."""

    n_failed: int
    n_completed: int
    interrupted: bool


def run_parallel_solve_batch(
    job_specs: list[ParallelSolveJob],
    max_workers: int,
    *,
    solutions_write_fn: Callable[[ParallelSolveJob, dict[str, Any]], Path],
    is_interrupted: Callable[[], bool],
) -> ParallelSolveBatchResult:
    """Execute ``job_specs`` with up to ``max_workers`` worker processes.

    Supported ``solver_name`` values: ``cudaq``, ``cirq``, ``brute_force``,
    ``simulated_annealing``.

    Creates a ``multiprocessing.Manager`` queue for worker status updates and
    shuts it down when the batch finishes.

    Args:
        job_specs: List of job descriptions (status queue injected here). All
            specs must share the same ``solver_name``.
        max_workers: Cap on concurrent processes (clamped to ``len(job_specs)``).
        solutions_write_fn: Maps ``(job, payload)`` to output path; must write JSON.
        is_interrupted: Predicate for cooperative exit (SIGINT / stop flag).

    Returns:
        Counts of failures and completed writes, and whether the batch was cut
        short by interrupt or cancel.

    """
    from experiments import cudaq_parallel as _cqp_mod

    ProcessPoolExecutor = _cqp_mod.ProcessPoolExecutor

    if not job_specs:
        return ParallelSolveBatchResult(n_failed=0, n_completed=0, interrupted=False)

    workers = min(max(1, max_workers), len(job_specs))
    solver_tag = job_specs[0].solver_name
    if any(s.solver_name != solver_tag for s in job_specs):
        raise ValueError("job_specs must share the same solver_name for one batch")
    status_prefix = f"[parallel {solver_tag}]"

    stop_event = threading.Event()
    running: set[str] = set()
    lock = threading.Lock()
    n_finished = 0
    total = len(job_specs)

    with mp.Manager() as manager:
        status_queue = manager.Queue()
        jobs = [replace(s, status_queue=status_queue) for s in job_specs]

        def pump_queue() -> None:
            try:
                while True:
                    kind, label = status_queue.get_nowait()
                    with lock:
                        if kind == "start":
                            running.add(label)
                        elif kind == "done":
                            running.discard(label)
            except Empty:
                pass

        def display_loop() -> None:
            """Refresh only when running set or write count changes (no 0.5s spam)."""
            last_snapshot: tuple[frozenset[str], int] | None = None
            while not stop_event.is_set():
                pump_queue()
                with lock:
                    snap = (frozenset(running), n_finished)
                if snap != last_snapshot:
                    last_snapshot = snap
                    cols = max(48, shutil.get_terminal_size(fallback=(100, 24)).columns)
                    line = _compact_parallel_status_line(
                        status_prefix, snap[0], snap[1], total, cols
                    )
                    print(f"\r\033[K{line}", end="", flush=True)
                stop_event.wait(0.05)

        display_thread = threading.Thread(
            target=display_loop, name="solver-parallel-display", daemon=True
        )
        display_thread.start()

        n_failed = 0
        n_completed = 0
        interrupted = False
        ctx = mp.get_context("spawn")
        executor: ProcessPoolExecutor | None = None
        try:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            pending: set[Future[tuple[int, dict[str, Any]]]] = set()
            future_to_job: dict[Future[tuple[int, dict[str, Any]]], ParallelSolveJob] = {}
            for job in jobs:
                fut = executor.submit(_parallel_solve_one_worker, job)
                pending.add(fut)
                future_to_job[fut] = job

            while pending:
                done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                pump_queue()
                for fut in done:
                    pending.discard(fut)
                    job = future_to_job[fut]
                    try:
                        _k, payload = fut.result()
                    except CancelledError:
                        continue
                    except Exception:
                        logger.exception(
                            "Worker future failed for instance %s", job.instance_json_path
                        )
                        payload = _payload_from_future_failure(job, traceback.format_exc())

                    if "error" in payload.get("solver_output", {}):
                        n_failed += 1
                    try:
                        solutions_write_fn(job, payload)
                    except Exception:
                        logger.exception("Failed writing solution for k=%s", job.k)
                        n_failed += 1
                    with lock:
                        n_finished += 1
                    n_completed += 1
                    gc.collect()

                if is_interrupted():
                    interrupted = True
                    break

            if interrupted and executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        finally:
            stop_event.set()
            pump_queue()
            display_thread.join(timeout=2.0)
            print("", flush=True)
            if executor is not None and not interrupted:
                executor.shutdown(wait=True)

    return ParallelSolveBatchResult(
        n_failed=n_failed,
        n_completed=n_completed,
        interrupted=interrupted,
    )
