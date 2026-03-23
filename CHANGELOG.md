# Changelog

All notable changes to this project are documented in this file.

---

## v0.1.0 (unreleased)

Initial research scaffold for Hotel TSP optimization with Tensor-QUDO and QUBO
formulations. Development period: March 7 -- March 21, 2026. 65 commits across
6 merged pull requests.

### Core formulations

- Tensor-QUDO formulation (`ProblemTQUDO`) with 3D `Etab` and 4D `Ettprimeab`
  tensors, using qudit encoding that inherently enforces one-city-per-timestep
  without `lambda_0`.
- QUBO formulation (`ProblemQUBO`) with symmetric quadratic matrix, binary
  one-hot encoding, and all three penalty terms (`lambda_0`, `lambda_1`,
  `lambda_2`).
- Energy normalisation for both formulations (max absolute value scaled to 1).
- QUBO-to-Ising conversion in `math_utils/qubo_ising.py`.

### Solver backends

- **Cirq** (3 formulation modes):
  - QUBO via Ising mapping with Rz/ZZ cost gates and Rx mixer.
  - TQUDO native with custom d-dimensional qudit gates:
    `QuditHadamardGate` (DFT), `QuditDiagonalCostGate` (diagonal cost
    encoding), `QuditRingMixerGate` (ring mixer).
  - TQUDO virtual (qubit emulation) with multi-controlled phase gates.
- **CUDA-Q** (GPU-accelerated, 2 formulation modes):
  - QUBO via SpinOperator with CNOT-Rz-CNOT decomposition.
  - TQUDO virtual (qubit emulation).
  - GPU target probing with fail-fast when no NVIDIA GPU is available.
- **Simulated Annealing** (classical, 2 formulation modes):
  - Permutation-based with three neighbourhood operators (swap, insert,
    2-opt reverse).
  - Geometric cooling schedule with configurable parameters.
- Shared `BaseQAOASolver` base class for Cirq and CUDA-Q.
- `SolverProtocol` interface with `SolverRunConfig` and `SolverResult`.
- TQA (Trotterized Quantum Annealing) parameter initialisation.
- Five scipy optimisers: COBYLA, Powell, L-BFGS-B, SLSQP, Nelder-Mead.

### Noise simulation

- Backend-agnostic `NoiseConfig` with 5 noise types: depolarizing,
  amplitude_damping, phase_damping, bit_flip, phase_flip.
- Per-gate probability overrides via `gate_noise` dict.
- Cirq: qubit noise via `ConstantQubitNoiseModelWithOverrides`, native qudit
  noise via d-dimensional Kraus channels.
- CUDA-Q: noise model with automatic density-matrix target selection.
- Environment kill-switch (`HTSP_ENABLE_NOISE_SIMULATION`).
- Memory scaling warnings for large systems.

### Instance generation

- Random instance generator with configurable city count, precedence ranges,
  and price ranges.
- Acyclic precedence graph enforcement via BFS cycle detection.
- Master seed with per-instance sub-seeds for reproducibility.
- YAML-based configuration (`config.yaml`, `solver_config.yaml`).
- Cross-validation of instance and solver compatibility.

### Experiment workflow

- `run_workflow()` orchestrating generation, solving, and JSON output.
- Incremental per-instance JSON saves to `output/raw/`.
- SIGINT handling (graceful interruption after current instance).
- Error recovery with traceback capture in output records.
- Progress reporter with single-line display and energy tracking.
- CLI entry point with `--instance-config`, `--solver-config`, `--output`.

### Infrastructure

- Editable package install via `pyproject.toml` with optional extras
  (dev, cirq, cudaq, ui).
- `install.sh` and `bin/setup.sh` for automated environment setup.
- `scripts/makefile` with setup, lint, test, app, and clean targets.
- Settings loaded from `.env` with `HTSP_*` prefix variables.
- ruff linting (line-length 100, Python 3.11 target; runtime 3.11–3.13).
- 17 test files with shared fixtures in `conftest.py`.
- GPU tests auto-skip when no NVIDIA GPU is available.
- Streamlit UI scaffold.

### Documentation

- `docs/formulations.md`: mathematical formulations with LaTeX equations and
  Cirq native qudit gate documentation.
- `docs/architecture.md`: technical architecture with module map, data flow,
  solver matrix, QAOA circuit details, noise system.
- `docs/api_reference.md`: comprehensive API reference for all public modules.
- `docs/configuration.md`: all configuration surfaces with tuning guidance.
- `docs/development.md`: development guide with test suite documentation.

### Project timeline

| Date       | Milestone                                                         |
|------------|-------------------------------------------------------------------|
| Mar 7      | First commit: project scaffolding and README                      |
| Mar 7-8    | Core formulations, cost functions, constraints validator           |
| Mar 8-9    | QAOA TQUDO implementation, SA and Cirq backends                   |
| Mar 9-10   | Experiment workflow, SA neighbourhood operators, QUBO-to-Ising fixes |
| Mar 10-11  | TQA initialisation, installation improvements, formulation corrections |
| Mar 11-13  | CUDA-Q GPU support, improved TQUDO configuration and shot types   |
| Mar 13-14  | Cirq native qudit implementation, sampling-based cost evaluation  |
| Mar 14-15  | Lambda corrections, SA parameter exposure, QUBO/TQUDO offset documentation |
| Mar 15-16  | Native qudit noise, per-gate noise overrides, noise model documentation |
| Mar 16-18  | Energy normalisation, sample storage, per-instance seeds, progress bar |
| Mar 18-20  | CUDA-Q noise fix, performance improvements, validation hardening  |
| Mar 20-21  | Circuit reuse optimisation, vectorised cost computation, formulation naming (`tqudo` vs `tqudo_virtual`), centralised QAOA runners |
