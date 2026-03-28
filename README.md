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
- Four solver backends: Cirq (native qudits + qubit emulation), CUDA-Q (GPU-accelerated), Simulated Annealing, and **brute force** (exact enumeration over the full QUBO / TQUDO assignment space within documented size limits).
- Noise simulation framework with 5 noise types and per-gate overrides.
- Experiment workflow with incremental JSON output and SIGINT handling; disk layout under `output/raw/solutions/<solver>/...`.
- **Data analysis** pipeline (`src/data_analysis/`): ingest → `output/processed/` metrics → `processed/plots_data/` → figures in `output/images/` (optional extra `analysis`: pandas, pyarrow, matplotlib, scipy).
- **Static results dashboard** ([`webpage_results/`](webpage_results/)): reads summaries and metrics from `output/processed/` over HTTP (not `file://`); use `make -f scripts/makefile results-web` after running the analysis pipeline.
- Test suite covering models, formulations, solvers, constraints, brute force, and data ingest.
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
`./install.sh dev,ui,cirq`. For the post-processing CLI (`data_analysis`), add
`analysis` (e.g. `./install.sh dev,ui,cudaq,analysis`). Installer validates `git` and `Python 3.11–3.13`
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
# After `pip install -e '.[analysis]'`:
make -f scripts/makefile analysis-all    # ingest + metrics + plots → processed/ + images/
# Serve repo root over HTTP for the static dashboard (fetch needs HTTP):
make -f scripts/makefile results-web     # → http://localhost:8765/webpage_results/index.html
```

## Project layout

```text
bin/                Bootstrap scripts (setup.sh)
docs/               Architecture, guides (repository, parallelism, experiments, analysis), API reference, configuration
input/              Optional input datasets
output/             Local results (ignored in git except placeholders)
scripts/            Task runner (makefile)
webpage_results/    Static HTML dashboard (metrics from output/processed/; open via HTTP)
src/
  config/           Runtime settings and environment loading (.env)
  data_analysis/    Ingest manifest, paired metrics vs brute_force ref, plots
  experiments/      Main experiment workflow (generate, solve, save)
  instance_gen_process/
                    Instance configuration, loading, generation, formulation builders
  math_utils/       QUBO-to-Ising conversion
  solvers/          Solver protocol + Cirq, CUDA-Q, SA, and brute_force
  streamlit_app/    Streamlit UI shell
  utils/            Costs (incl. batch), constraints, JSON/experiment serialisation,
                    YAML + disk path helpers, QAOA helpers, progress, output paths, logging
tests/              Pytest suite: smoke, contracts, unit tests, brute_force, data_analysis
```

## Documentation

### Guides

| Document | Description |
|----------|-------------|
| [docs/repository_guide.md](docs/repository_guide.md) | How the repo fits together: problem, `src/` layout, config → instances → solutions → analysis |
| [docs/parallelism_and_vectorization.md](docs/parallelism_and_vectorization.md) | Cross-instance process pools vs NumPy batching in brute force / `costs_batch` |
| [docs/extending_quantum_solvers.md](docs/extending_quantum_solvers.md) | Reusing parallel workers and `SolverProtocol` for other variational algorithms (e.g. VQE) |
| [docs/parallel_workers_general_workloads.md](docs/parallel_workers_general_workloads.md) | The same worker pattern for DL, sweeps, and other CPU/GPU batch jobs |
| [docs/experiments_design_and_artifacts.md](docs/experiments_design_and_artifacts.md) | Workflow modes, why JSON artifacts look like they do, top-level solution schema |
| [docs/analysis_and_figures.md](docs/analysis_and_figures.md) | Short reference: analysis commands, processed outputs, figure filenames |
| [docs/reproducing_results_from_scratch.md](docs/reproducing_results_from_scratch.md) | Command checklist from install through experiments, analysis, and static dashboard |

### Reference

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Deep technical architecture: module map, data flow, solver matrix, QAOA circuits, noise system, cost pipeline |
| [docs/formulations.md](docs/formulations.md) | Mathematical formulations (Tensor-QUDO and QUBO) with LaTeX equations, Cirq qudit gate documentation |
| [docs/api_reference.md](docs/api_reference.md) | API reference: models, generators, solvers (incl. brute force), utils, config, experiments, data_analysis |
| [docs/configuration.md](docs/configuration.md) | All configuration surfaces: `.env`, `config.yaml`, `solver_config.yaml`, compatibility rules, tuning guidance |
| [docs/development.md](docs/development.md) | Development guide: setup, testing, linting, conventions, parallel instance settings |
| [docs/data_analysis.md](docs/data_analysis.md) | Data analysis pipeline: manifest fields, metrics, aggregates, full figure catalog |
| [CHANGELOG.md](CHANGELOG.md) | Project history and release notes |

## Configuration

1. Copy `.env.example` to `.env` (done automatically by `install.sh`).
2. Adjust backend/output settings if needed.
3. Update `src/instance_gen_process/config.yaml` for instance generation.
4. Update `src/instance_gen_process/solver_config.yaml` for solver and QAOA settings.

See [docs/configuration.md](docs/configuration.md) for full reference.

## Output policy

- `output/raw/`: instance JSON under `raw/instances/...` and solution JSON under `raw/solutions/<solver>/<formulation>/n_<n>/...`.
- `output/processed/`: tables from `data_analysis` (`manifest.parquet`, `paired_metrics.parquet`, `summary_by_config.csv`, optional `energy_curves_agg.parquet`/`.csv`; SA rows are excluded from summaries and energy aggregates) plus `processed/plots_data/` (per-figure Parquet inputs produced by `data_analysis.prepare_plots`).
- `output/images/`: PNG figures from `data_analysis.plot`, which renders from `processed/plots_data/` (subfolders: `energy_history/`, `dashboards/`, `approx_ratio/`, `steps/`, `improvement/`, `p_opt/`) and writes `extended/` from `processed/*.parquet` (see `docs/data_analysis.md`).

Placeholders may be committed; bulk generated files are typically gitignored.
