# Development Guide

---

## Prerequisites

- **Python 3.12+** (required).
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

1. Verifies `python3` is available and >= 3.12.
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

| Extra   | Packages                                |
|---------|-----------------------------------------|
| `dev`   | pytest >= 8.0, ruff >= 0.6.0            |
| `cirq`  | cirq >= 1.3.0, scipy >= 1.10.0         |
| `cudaq` | cudaq >= 0.8.0, scipy >= 1.10.0        |
| `ui`    | streamlit >= 1.28.0                     |
| `all`   | All of the above                        |

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
```

### Running individual tests

```bash
.venv/bin/python -m pytest tests/test_costs.py -v
.venv/bin/python -m pytest tests/test_costs.py::test_name -v
```

### Running the experiment workflow

```bash
.venv/bin/python -m experiments.main_experiment_workflow
.venv/bin/python -m experiments.main_experiment_workflow --instance-config path/to/config.yaml
.venv/bin/python -m experiments.main_experiment_workflow --solver-config path/to/solver_config.yaml
.venv/bin/python -m experiments.main_experiment_workflow --output path/to/output
```

---

## Linting

**Linter**: ruff (configured in `pyproject.toml`).

```
[tool.ruff]
line-length = 100
target-version = "py312"
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
- **Python target**: 3.12+ (uses `X | Y` union syntax, `tuple[...]` generics).
- **No secrets in code**: `.env` is gitignored; `.env.example` is committed
  with safe defaults.

---

## Test suite

17 test files covering models, formulations, solvers, constraints, costs,
configuration, and imports.

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
├── tests/                # 17 test files + conftest.py
└── output/               # Experiment results (gitignored contents)
```
