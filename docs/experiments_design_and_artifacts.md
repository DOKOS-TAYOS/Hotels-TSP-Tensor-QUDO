# Experiments: design and saved artifacts

Why the workflow looks the way it does, what each mode does, and what lands on disk.

---

## Why JSON instances and solutions

- **Reproducibility:** Each solution file embeds a snapshot of the instance, merged solver config, and solver output so runs are self-describing.
- **Incremental IO:** Large grids write **per-instance** files so completed work survives interruption and partial reruns are easier than one monolithic log.
- **Downstream analysis:** `data_analysis` scans `raw/solutions/**/*.json` and does not depend on solver internals — only on stable top-level keys.

---

## Workflow entrypoint

CLI: `python -m experiments.main_experiment_workflow` (from the venv, with `PYTHONPATH` via editable install).

`--mode` is **required**. Common modes:

| Mode | Behavior |
|------|----------|
| `generate` | Builds instances from instance generation config + `config.yaml`; writes `raw/instances/n_<n>/instance_<k>.json`. |
| `cudaq` | Runs preset CUDA-Q experiment YAMLs (see `PRESET_EXPERIMENT_YAMLS` in [main_experiment_workflow.py](../src/experiments/main_experiment_workflow.py)). |
| `sa` | Preset simulated-annealing YAMLs. |
| `cirq5` | Preset Cirq TQUDO `n=5` experiment. |
| `brute_force` | Preset brute-force YAMLs (exact reference costs within size limits). |
| `experiment` | One or more YAMLs via `--experiment-yaml`; each merged with `solver_config.yaml`. |
| `check_feasibility` | Validates existing solution JSON for a chosen solver; see [development.md](development.md). |

Experiment YAMLs live in [src/experiments/](../src/experiments/) (e.g. `experiment_cudaq_n5_qubo.yaml`, `experiment_sa_n6_tqudo.yaml`). Run `generate` before modes that expect on-disk instances.

**Design intent:** Preset modes encode the paper-style benchmark matrix (formulations × backends × sizes); `experiment` mode is the escape hatch for custom grids.

---

## Output paths

Under the output root (default from `HTSP_OUTPUT_DIR`, often `output/`):

- **Instances:** `raw/instances/n_<n_cities>/instance_<k>.json`
- **Solutions:** `raw/solutions/<solver>/<formulation>/n_<n_cities>/[<qaoa_depth>/]instance_<k>.json`

`qaoa_depth` appears as a path segment for QAOA runs; simulated annealing typically omits it.

---

## What each solution JSON contains

Assembled by [build_solution_record](../src/utils/experiment_serialize.py):

| Key | Meaning |
|-----|---------|
| `instance` | Serialized `ProblemInstance` (prices, precedences, etc.). |
| `instance_config` | Instance-generation parameters for that run. |
| `instance_index` | Zero-based index in the batch (often `k-1` for instance `k`). |
| `solver_config` | Merged solver settings (formulation, QAOA depth, shots, noise, **effective parallel counts**, …). |
| `solver_output` | `serialize_solver_result`: `solver_name`, `objective_value`, `feasible`, `runtime_seconds`, `metadata` (and `error` if failed). |
| `instance_source` | Path to the input instance JSON when solved from disk. |

Typical `metadata` keys (when present): `energy_history`, `initial_energy`, `real_cost`, `best_sequence`, `best_bitstring`, `final_samples`, `configs_evaluated` (brute force), etc. Full ingest field list: [data_analysis.md](data_analysis.md).

**Brute-force runs** provide exact optima (within enumeration limits) so quantum / heuristic runs can be scored with `approx_ratio_*` in analysis.

---

## Parallelism and logs

When instance batches use multiple workers, CUDA-Q may redirect native stderr per [configuration.md](configuration.md) (`HTSP_SILENCE_NATIVE_STDERR`, log path under output root).

---

## Related

- [parallelism_and_vectorization.md](parallelism_and_vectorization.md)
- [reproducing_results_from_scratch.md](reproducing_results_from_scratch.md)
