# Extending with other quantum algorithms (e.g. VQE)

This codebase is built around **QAOA** for the Hotel TSP, but the **experiment harness** and **parallel worker** layer are intentionally separable. **VQE and other algorithms are not implemented** here; the following describes how you would plug them in or reuse the machinery.

---

## What you can reuse as-is

1. **Parallel batch driver** — [run_parallel_solve_batch](../src/experiments/parallel_solve_batch.py) with picklable `ParallelSolveJob` specs: spawn-based `ProcessPoolExecutor`, parent status line, cooperative interrupt, `solutions_write_fn` to persist JSON.
2. **Worker shape** — `_parallel_solve_one_worker`: load instance JSON, call a solver entry point, wrap output in `build_solution_record` ([utils/experiment_serialize.py](../src/utils/experiment_serialize.py)).
3. **Disk layout** — [workflow_io.py](../src/experiments/workflow_io.py) paths under `raw/solutions/<solver>/<formulation>/n_<n>/...` so `data_analysis` can ingest without changes (if you keep the same top-level JSON schema).
4. **Classical optimization loop pattern** — [solvers/_qaoa_base.py](../src/solvers/_qaoa_base.py) and CUDA-Q/Cirq entry points show how SciPy (or similar) drives variational parameters; a VQE would replace the expectation / circuit layer, not necessarily this file wholesale.

---

## Integrating a new backend cleanly

**Recommended path:** implement a class that satisfies [SolverProtocol](../src/solvers/base.py):

- `solver_name: str`
- `solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult`

Map quantum-specific options through `SolverRunConfig` (you may need to extend the dataclass and YAML loader if you add fields) and return `SolverResult` with `objective_value`, `feasible`, `runtime_seconds`, and a `metadata` dict (`energy_history`, `real_cost`, samples, etc.) consistent with what you want downstream.

**Wire parallel workers:** add your backend to `_PARALLEL_SOLVER_IMPORTS` in [parallel_solve_batch.py](../src/experiments/parallel_solve_batch.py) (module path + class name). Workers must be importable after `spawn`.

**Wire the CLI workflow:** extend [main_experiment_workflow.py](../src/experiments/main_experiment_workflow.py) wherever solver names are dispatched (preset modes, validation with `validate_solver_instance_compatibility` in [instance_gen_process](../src/instance_gen_process/), and [solver_config.yaml](../src/instance_gen_process/solver_config.yaml) parsing). This is the largest integration surface.

---

## VQE-specific notes

- **Circuit + observable:** Unlike QAOA’s alternating cost/mixer layers, VQE needs a problem Hamiltonian (or equivalent observable) and a parameterized ansatz. You would add modules under e.g. `solvers/my_vqe_solver/` mirroring the structure of `cudaq_solver/` (kernel/build, expectation from shots or statevector, parameter loop).
- **Formulation:** The existing instance types are QUBO / TQUDO for *this* combinatorial problem; your VQE would either consume the same `ProblemQUBO.qubo_matrix` / TQUDO tensors or map from `ProblemInstance` to a Pauli sum.
- **GPU vs CPU:** CUDA-Q and similar stacks typically require **one process per GPU context**. The current design matches that: one worker process owns one solve. For CPU-only frameworks (Cirq without GPU), the same pool pattern applies; cap workers like any CPU job.
- **Other frameworks:** If you do not use `SolverProtocol`, you can still copy the **pattern** (`ParallelSolveJob`-like spec + top-level worker + `ProcessPoolExecutor`) and write your own JSON envelope, but you must either keep the fields `data_analysis` expects or fork the analysis pipeline.

---

## Minimal “fork only the worker” approach

If you only want multi-GPU / multi-process **without** integrating YAML:

1. Copy `ParallelSolveJob` and `run_parallel_solve_batch` (or import them).
2. Replace `_parallel_solve_one_worker` body with your algorithm’s entry point, still producing a dict compatible with your writer.
3. Implement your own `solutions_write_fn` if you do not need the project’s JSON schema.

See also [parallel_workers_general_workloads.md](parallel_workers_general_workloads.md) for the same idea outside quantum.
