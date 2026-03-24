# Development Guide

---

## Prerequisites

- **Python 3.11, 3.12, or 3.13** (required).
- **Linux** (required). CUDA-Q and the install/task scripts require a Linux
  environment. WSL2 on Windows works for development.
- **NVIDIA GPU** (optional). Required only for the CUDA-Q backend. Tests
  auto-skip when no GPU is available.
- **Git** (required). The install script validates this before proceeding.

---

## Setup

### Quick start

```bash
./install.sh
```

This runs `bin/setup.sh` which:

1. Verifies `python3` is available and is 3.11, 3.12, or 3.13.
2. Creates a `.venv` virtual environment if missing.
3. Upgrades pip.
4. Installs the package in editable mode with the specified extras.
5. Copies `.env.example` to `.env` if `.env` does not exist.

### Extras selection

The default extras are `dev,ui,cudaq`. To use a different set:

```bash
./install.sh dev,ui,cirq       # Cirq backend instead of CUDA-Q
./install.sh dev               # Minimal: tests + linting only
./install.sh all               # Everything (requires GPU for CUDA-Q)
```

Available extras defined in `pyproject.toml`:

| Extra     | Packages                                                |
|-----------|---------------------------------------------------------|
| `dev`     | pytest >= 8.0, ruff >= 0.6.0                            |
| `cirq`    | cirq >= 1.3.0, scipy >= 1.10.0                         |
| `cudaq`   | cudaq >= 0.8.0, scipy >= 1.10.0                        |
| `ui`      | streamlit >= 1.28.0                                    |
| `analysis`| pandas >= 2.0, pyarrow >= 14, matplotlib >= 3.8, scipy  |
| `all`     | All of the above (including analysis)                  |

Core dependencies (always installed): `pyyaml >= 6.0`, `numpy >= 1.24.0`.

---

## Make targets

All targets use the virtual environment at `.venv/`:

```bash
make -f scripts/makefile setup   # Run bin/setup.sh with EXTRAS (default: dev,ui,cudaq)
make -f scripts/makefile lint    # ruff check .
make -f scripts/makefile test    # pytest (all tests)
make -f scripts/makefile app     # streamlit run src/streamlit_app/app.py
make -f scripts/makefile clean   # Remove caches, __pycache__, .pyc, temp dirs
# Require pip install -e '.[analysis]' for the following:
make -f scripts/makefile analysis-ingest   # raw JSON → processed/manifest.*
make -f scripts/makefile analysis-metrics  # manifest → paired_metrics, summary, curves, wilcoxon
make -f scripts/makefile analysis-plots    # processed tables → output/images/*.png
make -f scripts/makefile analysis-all      # ingest + metrics + plots (default --output-root output)
```

### Running individual tests

```bash
.venv/bin/python -m pytest tests/test_costs.py -v
.venv/bin/python -m pytest tests/test_costs.py::test_name -v
```

### Running the experiment workflow

Default **legacy** mode matches the old behaviour: load `config.yaml` and `solver_config.yaml`, generate instances in memory, write timestamped JSON files under `output/raw/`. Output root defaults to `HTSP_OUTPUT_DIR` (usually `output/`) when `--output` is omitted.

| `--mode` | What runs |
|----------|-----------|
| `legacy` (default) | In-memory generation + solve; filenames `exp_<timestamp>_inst_<i>_...json` |
| `generate` | From `src/experiments/instance_generation_config.yaml` and ranges/seed in `config.yaml`; writes `raw/instances/n_<n_cities>/instance_<k>.json` |
| `cudaq` | Preset CUDA-Q experiment YAMLs under `src/experiments/` |
| `sa` | Preset simulated-annealing experiment YAMLs |
| `cirq5` | Preset Cirq TQUDO n=5 experiment |
| `brute_force` | Preset exact-enumeration YAMLs (`experiment_brute_force_n5_qubo.yaml`, `..._n5_tqudo.yaml`, `..._n9_tqudo.yaml`) → `raw/solutions/brute_force/...` |
| `experiment` | One or more YAMLs via `--experiment-yaml f1.yaml f2.yaml` (merged each with `solver_config.yaml`) |
| `check_feasibility` | Scan `raw/solutions/<solver>/**/*.json` for one backend; print paths that are not feasible (requires `--check-solver brute_force|cudaq|cirq|simulated_annealing`). Exit 0 if all OK, 1 if any bad, 2 if the solver folder is missing |

Experiment modes read instances from `raw/instances/...` and write to `raw/solutions/<solver>/<formulation>/n_<n_cities>/[<depth>/]instance_<k>.json`. Run `generate` before experiment modes that need on-disk instances.

#### CUDA-Q: parallel instances (experiment on-disk mode only)

When `solver: cudaq` and the merged experiment YAML sets `cudaq_max_parallel_instances` to an integer **greater than 1**, each batch of on-disk instances for a fixed `(n_cities, qaoa_depth)` is solved with multiple **processes** (`multiprocessing` **spawn**), one CUDA-Q context per process. QAOA inside each instance stays sequential; only **different instances** overlap. Other solvers (except Cirq; see below) and `cudaq` with this key unset or set to `1` keep the previous sequential behaviour (with the usual single-line progress).

- **YAML**: optional `cudaq_max_parallel_instances` (default `1`), merged like other top-level solver keys.
- **Environment**: `HTSP_CUDAQ_MAX_PARALLEL_INSTANCES` overrides the YAML value when set (non-empty string).
- **UI**: child processes do not print QAOA step bars; the parent prints one **compact** line (rewritten in place) with active instance indices and write progress, **only when that state changes** (e.g. `[parallel cudaq] active_inst=[1,2,3] writes=7/100`) so narrow terminals do not wrap and spam the log.
- **VRAM**: each worker uses its own GPU memory footprint; reduce the parallel count if you hit OOM.
- **Interrupt**: Ctrl+C cancels work not yet started; workers already running may continue until they finish.
- **Reproducibility**: solution JSON includes `cudaq_max_parallel_instances_effective` under `solver_config` for CUDA-Q runs.

#### Cirq: parallel instances (experiment on-disk mode only)

When `solver: cirq` and the merged experiment YAML sets `cirq_max_parallel_instances` to an integer **greater than 1**, each batch of on-disk instances for a fixed `(n_cities, qaoa_depth)` is solved with multiple **processes** (`multiprocessing` **spawn**), one Cirq solve per process. QAOA inside each instance stays sequential; only **different instances** overlap.

- **YAML**: optional `cirq_max_parallel_instances` (default `1`), merged like other top-level solver keys.
- **Environment**: `HTSP_CIRQ_MAX_PARALLEL_INSTANCES` overrides the YAML value when set (non-empty string).
- **UI**: same as CUDA-Q: child processes do not print QAOA step bars; the parent shows only the compact line with prefix `[parallel cirq]`.
- **CPU / RAM**: each worker runs a full Cirq stack; scale workers to available cores and memory. With `cirq_max_parallel_instances > 1`, set **`OMP_NUM_THREADS=1`** (and similarly limit MKL/OpenBLAS thread pools if applicable) so each process does not spawn many BLAS threads and oversubscribe the CPU.
- **Reproducibility**: solution JSON includes `cirq_max_parallel_instances_effective` under `solver_config` for Cirq runs.

```bash
.venv/bin/python -m experiments.main_experiment_workflow
.venv/bin/python -m experiments.main_experiment_workflow --mode generate
.venv/bin/python -m experiments.main_experiment_workflow --mode sa --output path/to/output
.venv/bin/python -m experiments.main_experiment_workflow --mode experiment --experiment-yaml path/to/exp.yaml
.venv/bin/python -m experiments.main_experiment_workflow --mode check_feasibility --check-solver cudaq
.venv/bin/python -m experiments.main_experiment_workflow --mode brute_force
.venv/bin/python -m experiments.main_experiment_workflow --instance-config path/to/config.yaml
.venv/bin/python -m experiments.main_experiment_workflow --solver-config path/to/solver_config.yaml
.venv/bin/python -m experiments.main_experiment_workflow --output path/to/output
```

### Post-processing and figures (`data_analysis`)

After runs exist under `output/raw/`, install the `analysis` extra and run:

```bash
.venv/bin/python -m data_analysis.ingest --output-root output
.venv/bin/python -m data_analysis.metrics --output-root output
.venv/bin/python -m data_analysis.plot --output-root output
.venv/bin/python -m data_analysis.pipeline --output-root output          # all three
.venv/bin/python -m data_analysis.metrics --output-root output --sample-quality  # slower: histogram feasible mass
```

Artifacts: `output/processed/manifest.parquet` (or `.csv`), `paired_metrics.*`, `summary_by_config.csv`, optional `energy_curves_agg.parquet`, `wilcoxon_sa_qubo_tqudo.json`; figures under `output/images/`.

---

## Linting

**Linter**: ruff (configured in `pyproject.toml`).

```
[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = ["pytest-cache-files-*", ".tmp"]
```

Run: `make -f scripts/makefile lint` or `.venv/bin/python -m ruff check .`

---

## Code conventions

- **Package root**: `src/` (configured via `tool.setuptools.package-dir` in
  `pyproject.toml`). Imports use bare module names: `from solvers.base import ...`.
- **Pytest pythonpath**: `["src"]` so tests import the same way.
- **All dataclasses** use `frozen=True, slots=True`.
- **Type hints** required in all function signatures.
- **Python target**: 3.11–3.13 (uses `X | Y` union syntax, `tuple[...]` generics).
- **No secrets in code**: `.env` is gitignored; `.env.example` is committed
  with safe defaults.

---

## Test suite

Tests cover models, formulations, solvers (including brute force), constraints, costs,
configuration, data ingest, and imports.

### Test files

| File                          | What it tests                                                              |
|-------------------------------|----------------------------------------------------------------------------|
| `test_imports.py`             | Smoke test that main packages import without error                         |
| `test_settings.py`            | `load_settings()` from `.env` files and defaults                           |
| `test_instance_config.py`     | `load_instance_config()` validation, n_cities limits, precedence ranges    |
| `test_solver_config.py`       | `load_solver_config()` numeric validation, COBYLA budget, compatibility    |
| `test_qubo_tqudo_generation.py` | Penalty behaviour, QUBO vs TQUDO consistency, config validation          |
| `test_costs.py`               | QUBO cost `x^T Q x`, TQUDO cost from sequence, real cost, length checks   |
| `test_constraints.py`         | TQUDO/QUBO validation, precedence, duplicates, binary/sequence conversion  |
| `test_qubo_to_ising.py`       | `qubo_to_ising()` energy preservation, non-symmetric rejection             |
| `test_solver_contracts.py`    | Solver protocol compliance, formulation compatibility, CUDA-Q GPU check    |
| `test_cirq_tqudo.py`          | Qudit gates (Hadamard, diagonal cost, ring mixer), circuit build, run_qaoa |
| `test_cudaq_tqudo.py`         | CUDA-Q TQUDO kernel scaling, CPU simulator override                        |
| `test_cudaq_endianness.py`    | QUBO/TQUDO bitstring decoding vs `cudaq.sample`, noise target selection    |
| `test_cudaq_target.py`        | Target selection (nvidia, density-matrix-cpu), idempotency                 |
| `test_noise_models.py`        | NoiseConfig, Cirq/CUDA-Q noise, solver config round-trip, qudit Kraus      |
| `test_streamlit_app.py`       | Lazy import of Streamlit                                                   |
| `test_brute_force_solver.py`  | Exhaustive QUBO/TQUDO enumeration, config caps                             |
| `test_data_analysis.py`       | Manifest path parsing, ingest smoke tests                                |

### Shared fixtures (`conftest.py`)

| Fixture / helper              | Description                                                                |
|-------------------------------|----------------------------------------------------------------------------|
| `workspace_tmp_dir(prefix)`   | Creates isolated temp directory under `tests/.tmp`                         |
| `cleanup_workspace_tmp_dir()` | Removes temp directory and parent if empty                                 |
| `make_problem_instance()`     | Factory with sensible defaults (n_cities=4, unit prices, zero self-loops)  |
| `synthetic_tqudo_tensors()`   | Sparse tensors exercising local and long-range phases                      |
| `high_penalty_restriction()`  | All lambdas = 1000 (constraint enforcement tests)                          |
| `zero_penalty_restriction()`  | All lambdas = 0 (isolate cost terms from penalties)                        |

### GPU auto-skip

Tests requiring CUDA-Q with GPU automatically skip when no NVIDIA GPU is
available. This is handled via try/except around `cudaq` imports and target
probing.

### Pytest configuration

```
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-p no:cacheprovider"
```

Cache provider is disabled (`-p no:cacheprovider`) to avoid polluting the
workspace with `.pytest_cache` directories.

---

## Branching conventions

- `dev_cudaq`: main development branch (current active branch).
- Feature work is done in topic branches and merged via pull requests.
- Commit only code, config, and docs. Do not commit generated artifacts
  (`output/raw/`, `output/processed/`, `output/images/`).
- Record solver settings and seeds alongside every benchmark run for
  reproducibility.

---

## Project structure summary

```
Hotels-TSP-Tensor-QUDO/
├── .env.example          # Environment variable template
├── pyproject.toml        # Package metadata, dependencies, tool config
├── install.sh            # Entry point for setup
├── bin/setup.sh          # Core setup logic
├── scripts/makefile      # Task runner targets
├── docs/                 # Project documentation
├── src/                  # Package source (8 sub-packages)
├── tests/                # pytest + conftest.py
└── output/               # Experiment results (gitignored contents)
```
