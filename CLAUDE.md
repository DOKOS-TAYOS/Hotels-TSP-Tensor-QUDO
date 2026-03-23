# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Research scaffold for combinatorial optimization of a Hotel TSP (travel routing with precedence constraints) using **Tensor-QUDO** and **QUBO** formulations with quantum/classical solver backends. Reference paper: arXiv:2508.01958.

**Platform:** Linux only. Python 3.12+.

## Common commands

```bash
# Setup (creates .venv, installs editable package)
./install.sh                    # default: dev,ui,cudaq extras
./install.sh dev,ui,cirq        # for Cirq backend instead

# Lint, test, UI, clean
make -f scripts/makefile lint   # ruff check .
make -f scripts/makefile test   # pytest
make -f scripts/makefile app    # streamlit run
make -f scripts/makefile clean

# Run a single test
.venv/bin/python -m pytest tests/test_costs.py -v
.venv/bin/python -m pytest tests/test_costs.py::test_name -v

# Run experiment workflow (default: legacy in-memory batch → timestamped JSON in output/raw/)
.venv/bin/python -m experiments.main_experiment_workflow
.venv/bin/python -m experiments.main_experiment_workflow --instance-config path/to/config.yaml
.venv/bin/python -m experiments.main_experiment_workflow --solver-config path/to/solver_config.yaml
.venv/bin/python -m experiments.main_experiment_workflow --output path/to/output

# Calibration CLIs (output to output/T0sampling/ and output/lambdasSampling/)
.venv/bin/python -m experiments.estimate_t0 --n-instances 5 --chi0 0.8
.venv/bin/python -m experiments.estimate_lambdas --formulation qubo --lambda-values 10,50,100,500,1000
# Modes: --mode generate | cudaq | sa | cirq5 | experiment | check_feasibility (see docs/development.md)
```

## Architecture

### Two formulations of the same problem

1. **Tensor-QUDO** (`ProblemTQUDO`): Uses qudits (d-dimensional). Cost encoded in `Etab[t,a,b]` (3D) and `Ettprimeab[t,t',a,b]` (4D) tensors. No `lambda_0` penalty needed — qudit encoding inherently enforces one-city-per-timestep.
2. **QUBO** (`ProblemQUBO`): Uses binary one-hot variables. Cost encoded in symmetric `qubo_matrix` (2D). Needs all three lambda penalties. Objective has a constant offset vs real cost: `QUBO_cost = real_cost - (lambda_0 + lambda_1) * n_available`.

Both formulations normalise their tensors/matrix by `energy_scale = max(|values|, 1.0)` so entries lie in `[-1, 1]`. Sampled costs must be multiplied back by `energy_scale` to recover original units. Use `utils.costs.calculate_real_cost()` for formulation-independent cost comparisons.

### Data flow

**Legacy (`--mode legacy`, default):** `config.yaml` + `solver_config.yaml` → generate instances in memory → solver → JSON in `output/raw/` (`exp_<timestamp>_...`).

**Disk workflow:** `src/experiments/instance_generation_config.yaml` + `config.yaml` → JSON instances under `output/raw/instances/n_<n_cities>/instance_<k>.json`. Each `src/experiments/experiment_*.yaml` merges with `solver_config.yaml` and writes solutions to `output/raw/solutions/<solver>/<formulation>/n_<n_cities>/[<qaoa_depth>/]instance_<k>.json` (no `qaoa_depth` folder for simulated annealing).

### Solver backends (all implement `SolverProtocol`)

| Backend | Module | Formulations | Requirement |
|---------|--------|-------------|-------------|
| **Cirq** | `solvers/cirq_solver/` | `qubo`, `tqudo` (native qudits), `tqudo_virtual` (qubit emulation) | `cirq` extra |
| **CUDA-Q** | `solvers/cudaq_solver/` | `qubo`, `tqudo_virtual` | `cudaq` extra + NVIDIA GPU |
| **Simulated Annealing** | `solvers/simulated_annealing/` | `qubo`, `tqudo` | no extra deps |

Three formulation values exist in `solver_config.yaml`: `qubo`, `tqudo` (native qudits — Cirq and SA only), `tqudo_virtual` (qubit emulation — Cirq and CUDA-Q). Incompatible combos are rejected by `validate_solver_instance_compatibility()`.

Each solver dispatches to formulation-specific QAOA circuit modules (e.g., `qaoa_circuit_qubo.py`, `qaoa_circuit_tqudo.py`). Shared QAOA logic (parameter init, optimization loop) lives in `solvers/_qaoa_base.py`.

### Key modules

- **`instance_gen_process/generator.py`**: `generate_TQUDO_from_problem()`, `generate_QUBO_from_problem()`, `generate_random_set_instances()`.
- **`instance_gen_process/models.py`**: Core dataclasses: `ProblemInstance`, `ProblemTQUDO`, `ProblemQUBO`, `RestrictionConfig`, `InstanceConfig`.
- **`solvers/base.py`**: `SolverProtocol`, `SolverRunConfig`, `SolverResult`.
- **`solvers/_qaoa_base.py`**: Shared QAOA solver logic (parameter init via TQA, SciPy optimization loop) used by both Cirq and CUDA-Q backends.
- **`solvers/noise.py`**: `NoiseConfig` — backend-agnostic noise config consumed by both Cirq and CUDA-Q.
- **`utils/costs.py`**: `calculate_qubo_cost()`, `calculate_tqudo_cost()`, `calculate_real_cost()`.
- **`utils/constraints.py`**: Validation helpers, binary/sequence conversion, cycle detection.
- **`config/settings.py`**: `Settings` loaded from `.env` (prefix: `HTSP_*`). Noise kill-switch via `HTSP_ENABLE_NOISE_SIMULATION`.
- **`experiments/`**: CLI tools — `main_experiment_workflow.py` (full solve pipeline), `estimate_t0.py` (SA initial temperature via Ben-Ameur), `estimate_lambdas.py` (grid search over lambda penalties).

### Cirq native qudit gates (TQUDO)

The Cirq TQUDO backend uses three custom gates on `cirq.LineQid(dimension=d)`:
- `QuditHadamardGate`: d-dimensional DFT for uniform superposition.
- `QuditDiagonalCostGate`: diagonal 2-qudit gate encoding cost tensor slices.
- `QuditRingMixerGate`: `exp(-i*beta*(X_d + X_d†)/2)` — ring mixer (NOT equivalent to per-qubit Rx for d>2).

### Configuration files

- **`src/instance_gen_process/config.yaml`**: Instance generation (n_cities, price ranges, seed).
- **`src/instance_gen_process/solver_config.yaml`**: Solver choice, formulation, QAOA params, SA params, noise config.
- **`.env`**: Runtime environment (`HTSP_QUANTUM_BACKEND`, `HTSP_OUTPUT_DIR`, `HTSP_ENABLE_NOISE_SIMULATION`, etc.).

## Code conventions

- Package root is `src/` (configured in `pyproject.toml` via `tool.setuptools.package-dir`).
- `pythonpath = ["src"]` in pytest config — imports use bare module names (e.g., `from solvers.base import ...`).
- Linter: `ruff` with `line-length = 100`, `target-version = "py312"`.
- All dataclasses use `frozen=True, slots=True`.
- Type hints required in all function signatures. Uses Python 3.12+ syntax (`X | Y` unions, `tuple[...]` generics).
- CUDA-Q tests auto-skip when no GPU is available.
- Pytest runs with `-p no:cacheprovider` to avoid `.pytest_cache` pollution.
- Math reference for cost equations: `docs/formulations.md`.
