# Parallel workers for general workloads

The experiment code uses a small, reusable pattern: **picklable job specs**, **`multiprocessing` spawn**, and **`ProcessPoolExecutor`**. The same idea applies to **deep learning**, **hyperparameter search**, **simulations**, or any embarrassingly parallel batch where each task is heavy enough to justify a process.

---

## Pattern in this repo

1. Define a **frozen dataclass** (or plain structure) with everything the worker needs; it must be picklable under the `spawn` start method.
2. Use a **top-level function** as the worker entry point (required for spawn: no lambdas tied to local scope).
3. Create `multiprocessing.get_context("spawn")` and `ProcessPoolExecutor(max_workers=..., mp_context=ctx)`.
4. Submit jobs, collect futures, handle failures and interruption explicitly.

Reference: [run_parallel_solve_batch](../src/experiments/parallel_solve_batch.py) and [estimate_lambdas.py](../src/experiments/estimate_lambdas.py) (`_solve_instances_for_combo`).

---

## Applying it elsewhere

**Example shapes:**

- **DL training:** each job = `(seed, hyperparam_dict, data shard path)`; worker trains one model and writes a checkpoint + metrics JSON.
- **Cross-validation:** each job = `(fold_index, ...)`; worker fits and returns scores.
- **Simulation sweeps:** each job = parameter vector; worker runs simulation and appends one row to a result file (beware concurrent writes — prefer one file per job then merge).

You do **not** need to import this repository’s solver stack; you only reuse the **concurrency skeleton**.

---

## Caveats

| Topic | Guidance |
|-------|----------|
| **Pickling** | Spawn re-imports your module; large read-only blobs are better passed as paths and loaded inside the worker. |
| **CUDA / GPU** | Typically one context per process. Do not fork after CUDA init. Match `max_workers` to GPUs or accept device staging. |
| **BLAS / OpenMP** | With many CPU workers, set `OMP_NUM_THREADS=1` per process to avoid oversubscription (same recommendation as [development.md](development.md)). |
| **Memory** | Each worker duplicates heavy imports; fewer workers may be faster if RAM-bound. |
| **Stdio / logging** | Child processes can scramble progress bars; the repo’s experiment code uses a manager queue for compact parent status — optional for your app. |

**stdlib reference:** [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) — `ProcessPoolExecutor`, `wait`, cancellation.

---

## Relation to this project’s quantum batches

Same machinery as [parallelism_and_vectorization.md](parallelism_and_vectorization.md); if you extend quantum solvers, read [extending_quantum_solvers.md](extending_quantum_solvers.md).
