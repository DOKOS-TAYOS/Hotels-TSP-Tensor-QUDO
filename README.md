# Hotel TSP with Tensor-QUDO

Research scaffold for combinatorial optimization of a travel routing problem with
Tensor-QUDO formulations and multiple solver backends.

The problem: given a set of cities to visit (without repetition), find the route
that minimizes total cost. Costs include:
- **Travel cost**: cost of traveling on a given day (timestep) to a location (varies by day)
- **Hotel cost**: cost of staying in a city for that timestep

Each stay and each trip lasts one timestep. An additional constraint is **precedence**:
to be in city B, you must first have been in city A.

Formulation reference: [Introduction to QUDO, Tensor QUDO and HOBO formulations](https://arxiv.org/abs/2508.01958) (arXiv:2508.01958).
Cost equations: see [docs/formulations.md](docs/formulations.md).

**Platform:** Linux only. CUDA-Q and the project tooling are not supported on Windows or macOS. WSL2 works for development.

## Current status

**Version**: 0.1.0 (alpha, unreleased)

- Two mathematical formulations: Tensor-QUDO (native qudits) and QUBO (binary one-hot).
- Three solver backends: Cirq (native qudits + qubit emulation), CUDA-Q (GPU-accelerated), Simulated Annealing.
- Noise simulation framework with 5 noise types and per-gate overrides.
- Experiment workflow with incremental JSON output and SIGINT handling.
- 17 test files covering models, formulations, solvers, and constraints.
- Streamlit UI scaffold.

## Maintainers

- Adriano Lusso
- Alejandro Mata Ali

## Quickstart

```bash
./install.sh
```

Setup script creates `.venv`, installs editable project dependencies, and defaults to
the `dev,ui,cudaq` extras. For the Cirq backend instead, run
`./install.sh dev,ui,cirq`. Installer validates `git` and `Python 3.12+`
before running setup. The `cudaq` extra installs both CUDA-Q and SciPy.

## CUDA-Q backend contract

- `solver: cudaq` requires a Linux environment with a compatible NVIDIA GPU.
- CUDA-Q now fails fast when no NVIDIA GPU is available; it no longer falls back to CPU.
- CUDA-Q supports both `qubo` and `tqudo_virtual` formulations.

## Run common tasks

```bash
make -f scripts/makefile lint
make -f scripts/makefile test
make -f scripts/makefile app
make -f scripts/makefile clean
```

## Project layout

```text
bin/                Bootstrap scripts (setup.sh)
docs/               Architecture, formulations, API reference, configuration, development
input/              Optional input datasets
output/             Local results (ignored in git except placeholders)
scripts/            Task runner (makefile)
src/
  config/           Runtime settings and environment loading (.env)
  data_analysis/    Raw -> processed analysis pipeline (scaffold)
  experiments/      Main experiment workflow (generate, solve, save)
  instance_gen_process/
                    Instance configuration, loading, generation, formulation builders
  math_utils/       QUBO-to-Ising conversion
  solvers/          Solver protocol + Cirq, CUDA-Q, and SA backends
  streamlit_app/    Streamlit UI shell
  utils/            Costs, constraints, QAOA helpers, progress, output paths, logging
tests/              17 test files: smoke, contract, and unit tests
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Deep technical architecture: module map, data flow, solver matrix, QAOA circuits, noise system, cost pipeline |
| [docs/formulations.md](docs/formulations.md) | Mathematical formulations (Tensor-QUDO and QUBO) with LaTeX equations, Cirq qudit gate documentation |
| [docs/api_reference.md](docs/api_reference.md) | Comprehensive API reference for all public modules: models, generators, solvers, utils, config |
| [docs/configuration.md](docs/configuration.md) | All configuration surfaces: `.env`, `config.yaml`, `solver_config.yaml`, compatibility rules, tuning guidance |
| [docs/development.md](docs/development.md) | Development guide: setup, testing, linting, conventions, branching |
| [CHANGELOG.md](CHANGELOG.md) | Project history and release notes |

## Configuration

1. Copy `.env.example` to `.env` (done automatically by `install.sh`).
2. Adjust backend/output settings if needed.
3. Update `src/instance_gen_process/config.yaml` for instance generation.
4. Update `src/instance_gen_process/solver_config.yaml` for solver and QAOA settings.

See [docs/configuration.md](docs/configuration.md) for full reference.

## Output policy

- `output/raw`: intermediate experiment dumps (JSON per instance).
- `output/processed`: curated benchmark datasets (scaffold).
- `output/images`: figures for the paper (scaffold).

These folders remain in the repo as placeholders, but generated contents are ignored.
