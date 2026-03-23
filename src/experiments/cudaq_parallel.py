"""Parallel instance solves for on-disk experiment workflows (CUDA-Q and Cirq).

Uses ``multiprocessing`` spawn + :class:`~concurrent.futures.ProcessPoolExecutor`
so each worker owns its own backend context (CUDA-Q on GPU; Cirq on CPU).
Parent process shows which instance labels are currently running.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import multiprocessing as mp
import os
import re
import shutil
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from solvers.base import SolverResult, SolverRunConfig

from experiments.workflow_io import load_problem_instance_json, serialize_problem_instance

logger = logging.getLogger(__name__)

_INST_LABEL_RE = re.compile(r"\binst=(\d+)\s*$")

# Pool workers set this so ``utils.progress`` stays quiet (CUDA-Q and Cirq parallel batches).
EXPERIMENT_CUDA_WORKER_ENV = "HTSP_EXPERIMENT_CUDA_WORKER"
CUDAQ_PARALLEL_ENV = "HTSP_CUDAQ_MAX_PARALLEL_INSTANCES"
CIRQ_PARALLEL_ENV = "HTSP_CIRQ_MAX_PARALLEL_INSTANCES"


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


def resolve_cirq_max_parallel_instances(cfg_dict: dict[str, Any]) -> int:
    """Resolve parallel worker count: env overrides YAML ``cirq_max_parallel_instances``.

    Args:
        cfg_dict: Merged experiment + base solver mapping (may include
            ``cirq_max_parallel_instances``).

    Returns:
        Integer >= 1.
    """
    raw = os.environ.get(CIRQ_PARALLEL_ENV)
    if raw is not None and str(raw).strip() != "":
        w = int(raw)
    else:
        w = int(cfg_dict.get("cirq_max_parallel_instances", 1))
    return max(1, w)


def _metadata_to_json(obj: Any) -> Any:
    """Recursively normalise metadata for JSON (same rules as workflow)."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [_metadata_to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _metadata_to_json(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _metadata_to_json(dataclasses.asdict(obj))
    return obj


def _serialize_solver_result(result: SolverResult) -> dict[str, Any]:
    """Convert :class:`~solvers.base.SolverResult` to a JSON-friendly dict."""
    return {
        "solver_name": result.solver_name,
        "objective_value": result.objective_value,
        "feasible": result.feasible,
        "runtime_seconds": result.runtime_seconds,
        "metadata": _metadata_to_json(result.metadata),
    }


@dataclass(frozen=True, slots=True)
class CudaqParallelJobSpec:
    """Picklable description of one solve; queue is attached in the parent."""

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


@dataclass(frozen=True, slots=True)
class CudaqParallelJob:
    """Picklable unit of work for one on-disk instance (``cudaq`` or ``cirq``)."""

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
    status_queue: Any


def _cudaq_solve_one_worker(job: CudaqParallelJob) -> tuple[int, dict[str, Any]]:
    """Run CUDA-Q on one instance in a child process (top-level for spawn)."""
    from solvers import CudaqSolver

    os.environ[EXPERIMENT_CUDA_WORKER_ENV] = "1"
    q = job.status_queue
    q.put(("start", job.status_label))
    src = Path(job.instance_json_path)
    try:
        instance = load_problem_instance_json(src)
        result = CudaqSolver().solve(instance, job.run_config)
        payload: dict[str, Any] = {
            "instance": serialize_problem_instance(instance),
            "instance_config": job.instance_config_dict,
            "instance_index": job.k - 1,
            "instance_source": str(src),
            "solver_config": job.solver_config_serializable,
            "solver_output": _serialize_solver_result(result),
        }
        return job.k, payload
    except Exception:
        logger.exception("CUDA-Q worker failed for %s", src)
        try:
            instance = load_problem_instance_json(src)
            inst_dict = serialize_problem_instance(instance)
        except Exception:
            inst_dict = {}
        payload = {
            "instance": inst_dict,
            "instance_config": job.instance_config_dict,
            "instance_index": job.k - 1,
            "instance_source": str(src),
            "solver_config": job.solver_config_serializable,
            "solver_output": {
                "solver_name": job.solver_name,
                "error": traceback.format_exc(),
            },
        }
        return job.k, payload
    finally:
        q.put(("done", job.status_label))


def _cirq_solve_one_worker(job: CudaqParallelJob) -> tuple[int, dict[str, Any]]:
    """Run Cirq on one instance in a child process (top-level for spawn)."""
    from solvers import CirqSolver

    os.environ[EXPERIMENT_CUDA_WORKER_ENV] = "1"
    q = job.status_queue
    q.put(("start", job.status_label))
    src = Path(job.instance_json_path)
    try:
        instance = load_problem_instance_json(src)
        result = CirqSolver().solve(instance, job.run_config)
        payload: dict[str, Any] = {
            "instance": serialize_problem_instance(instance),
            "instance_config": job.instance_config_dict,
            "instance_index": job.k - 1,
            "instance_source": str(src),
            "solver_config": job.solver_config_serializable,
            "solver_output": _serialize_solver_result(result),
        }
        return job.k, payload
    except Exception:
        logger.exception("Cirq worker failed for %s", src)
        try:
            instance = load_problem_instance_json(src)
            inst_dict = serialize_problem_instance(instance)
        except Exception:
            inst_dict = {}
        payload = {
            "instance": inst_dict,
            "instance_config": job.instance_config_dict,
            "instance_index": job.k - 1,
            "instance_source": str(src),
            "solver_config": job.solver_config_serializable,
            "solver_output": {
                "solver_name": job.solver_name,
                "error": traceback.format_exc(),
            },
        }
        return job.k, payload
    finally:
        q.put(("done", job.status_label))


def _parallel_solve_one_worker(job: CudaqParallelJob) -> tuple[int, dict[str, Any]]:
    """Dispatch to CUDA-Q or Cirq worker (top-level for spawn pickling)."""
    if job.solver_name == "cudaq":
        return _cudaq_solve_one_worker(job)
    if job.solver_name == "cirq":
        return _cirq_solve_one_worker(job)
    raise ValueError(f"parallel batch unsupported for solver_name={job.solver_name!r}")


@dataclass(frozen=True, slots=True)
class CudaqParallelBatchResult:
    """Outcome of solving a batch of instances with a process pool."""

    n_failed: int
    n_completed: int
    interrupted: bool


def run_cudaq_parallel_batch(
    job_specs: list[CudaqParallelJobSpec],
    max_workers: int,
    *,
    solutions_write_fn: Callable[[CudaqParallelJob, dict[str, Any]], Path],
    is_interrupted: Callable[[], bool],
) -> CudaqParallelBatchResult:
    """Execute *job_specs* with up to *max_workers* worker processes (CUDA-Q or Cirq).

    Creates a :class:`multiprocessing.managers.SyncManager` for the status queue
    and shuts it down when the batch finishes.

    Args:
        job_specs: Non-empty list of job descriptions (queue added here). All
            specs must share the same ``solver_name`` (``cudaq`` or ``cirq``).
        max_workers: Cap on concurrent processes (clamped to ``len(job_specs)``).
        solutions_write_fn: Maps (job, payload) to output path; must write JSON.
        is_interrupted: Predicate for cooperative exit (SIGINT / stop flag).

    Returns:
        Counts of failures and completed writes, and whether the batch was cut
        short by interrupt/cancel.
    """
    if not job_specs:
        return CudaqParallelBatchResult(n_failed=0, n_completed=0, interrupted=False)

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
        jobs = [
            CudaqParallelJob(
                k=s.k,
                instance_json_path=s.instance_json_path,
                status_label=s.status_label,
                run_config=s.run_config,
                instance_config_dict=s.instance_config_dict,
                solver_config_serializable=s.solver_config_serializable,
                solver_name=s.solver_name,
                formulation=s.formulation,
                n_cities=s.n_cities,
                path_depth=s.path_depth,
                output_root=s.output_root,
                status_queue=status_queue,
            )
            for s in job_specs
        ]

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
            future_to_job: dict[Future[tuple[int, dict[str, Any]]], CudaqParallelJob] = {}
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
                        with lock:
                            n_finished += 1
                        n_failed += 1
                        n_completed += 1
                        continue

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

    return CudaqParallelBatchResult(
        n_failed=n_failed,
        n_completed=n_completed,
        interrupted=interrupted,
    )


run_parallel_instance_batch = run_cudaq_parallel_batch
