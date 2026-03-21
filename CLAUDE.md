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

# Run experiment workflow
.venv/bin/python -m experiments.main_experiment_workflow
```

## Architecture

### Two formulations of the same problem

1. **Tensor-QUDO** (`ProblemTQUDO`): Uses qudits (d-dimensional). Cost encoded in `Etab[t,a,b]` (3D) and `Ettprimeab[t,t',a,b]` (4D) tensors. No `lambda_0` penalty needed — qudit encoding inherently enforces one-city-per-timestep.
2. **QUBO** (`ProblemQUBO`): Uses binary one-hot variables. Cost encoded in symmetric `qubo_matrix` (2D). Needs all three lambda penalties. Objective has a constant offset vs real cost: `QUBO_cost = real_cost - (lambda_0 + lambda_1) * n_available`.

Use `utils.costs.calculate_real_cost()` for formulation-independent cost comparisons.

### Data flow

`config.yaml` + `solver_config.yaml` → `ProblemInstance` → formulation generation → solver → `SolverResult` → JSON in `output/raw/`

### Solver backends (all implement `SolverProtocol`)

| Backend | Module | Formulations | Requirement |
|---------|--------|-------------|-------------|
| **Cirq** | `solvers/cirq_solver/` | QUBO + TQUDO (native qudits & qubit emulation) | `cirq` extra |
| **CUDA-Q** | `solvers/cudaq_solver/` | QUBO + TQUDO | `cudaq` extra + NVIDIA GPU |
| **Simulated Annealing** | `solvers/simulated_annealing/` | QUBO + TQUDO | no extra deps |

Each solver dispatches to formulation-specific QAOA circuit modules (e.g., `qaoa_circuit_qubo.py`, `qaoa_circuit_tqudo.py`).

### Key modules

- **`instance_gen_process/generator.py`**: `generate_TQUDO_from_problem()`, `generate_QUBO_from_problem()`, `generate_random_set_instances()`.
- **`instance_gen_process/models.py`**: Core dataclasses: `ProblemInstance`, `ProblemTQUDO`, `ProblemQUBO`, `RestrictionConfig`, `InstanceConfig`.
- **`solvers/base.py`**: `SolverProtocol`, `SolverRunConfig`, `SolverResult`.
- **`solvers/noise.py`**: `NoiseConfig` — backend-agnostic noise config consumed by both Cirq and CUDA-Q.
- **`utils/costs.py`**: `calculate_qubo_cost()`, `calculate_tqudo_cost()`, `calculate_real_cost()`.
- **`utils/constraints.py`**: Validation helpers, binary/sequence conversion, cycle detection.
- **`config/settings.py`**: `Settings` loaded from `.env` (prefix: `HTSP_*`). Noise kill-switch via `HTSP_ENABLE_NOISE_SIMULATION`.

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
- CUDA-Q tests auto-skip when no GPU is available.
- Math reference for cost equations: `docs/formulations.md`.
