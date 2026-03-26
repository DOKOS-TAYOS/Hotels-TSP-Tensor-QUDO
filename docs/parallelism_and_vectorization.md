# Parallelism and vectorization

This repository uses **two distinct** mechanisms to speed up computation. They address different bottlenecks and should not be confused.

---

## 1. Cross-instance parallelism (experiment batches)

**Where:** [experiments/parallel_solve_batch.py](../src/experiments/parallel_solve_batch.py), invoked from [main_experiment_workflow.py](../src/experiments/main_experiment_workflow.py) when solving many on-disk instances.

**Idea:** Run **different instances** (different JSON inputs) in **different OS processes** using `multiprocessing` **spawn** and `concurrent.futures.ProcessPoolExecutor`. Each worker executes `_parallel_solve_one_worker`: it imports the solver class from `_PARALLEL_SOLVER_IMPORTS`, calls `solver.solve(instance, run_config)`, and writes a standard solution payload via `build_solution_record`.

**Supported `solver_name` values in the pool:** `cudaq`, `cirq`, `brute_force`, `simulated_annealing` (see `_PARALLEL_SOLVER_IMPORTS` in the same module).

**Configuration:**

- CUDA-Q: `cudaq_max_parallel_instances` in merged experiment YAML, overridden by `HTSP_CUDAQ_MAX_PARALLEL_INSTANCES` when set.
- CPU backends (`cirq`, `brute_force`, `simulated_annealing`): `cpu_max_parallel_instances` / `HTSP_CPU_MAX_PARALLEL_INSTANCES`.

**Important limits:**

- **QAOA inside one instance is still sequential** — only **different instances** overlap in time.
- **CUDA-Q:** each process holds its own GPU context and memory footprint; reduce parallel count if you hit OOM.
- **CPU:** set `OMP_NUM_THREADS=1` (and limit MKL/OpenBLAS threads if applicable) so many workers do not oversubscribe cores — see [development.md](development.md).

**Reproducibility:** effective parallel counts are recorded under `solver_config` in solution JSON (e.g. `cudaq_max_parallel_instances_effective`, `cpu_max_parallel_instances_effective`).

**Related:** calibration tool [estimate_lambdas.py](../src/experiments/estimate_lambdas.py) uses `ProcessPoolExecutor` only for certain **CPU** solvers across instances; CUDA-Q runs sequentially there.

---

## 2. Intra-routine vectorization (NumPy batches)

**Where:**

- [utils/costs_batch.py](../src/utils/costs_batch.py) — `batch_qubo_costs`, `batch_tqudo_costs`: vectorized objective for many assignments at once.
- [solvers/brute_force/solver.py](../src/solvers/brute_force/solver.py) — enumerates assignments in chunks (`_BRUTE_FORCE_ASSIGNMENT_CHUNK_SIZE`, default 8192), decodes bit patterns with `unpack_qubo_bitmatrix`, evaluates a batch of costs per chunk.

**Idea:** Within **one** brute-force solve, evaluate thousands of candidate assignments per NumPy operation instead of a Python loop per assignment.

**What this does *not* do:** it does **not** parallelize across CPU cores by itself (single process). It reduces per-assignment overhead. Cross-instance parallelism (above) is still the way to run many brute-force jobs concurrently.

---

## Summary

| Mechanism | Parallelizes | Typical use |
|-----------|----------------|-------------|
| Process pool (`run_parallel_solve_batch`) | Multiple instances / multiple processes | Large experiment grids on disk |
| `costs_batch` + brute-force chunks | Many assignments in one process | Fast exact search for one instance |

For reusing the process-pool pattern outside QAOA, see [extending_quantum_solvers.md](extending_quantum_solvers.md) and [parallel_workers_general_workloads.md](parallel_workers_general_workloads.md).
