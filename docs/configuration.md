# Configuration Reference

All configuration surfaces for the Hotels-TSP-Tensor-QUDO project.

---

## Environment variables (`.env`)

Copy `.env.example` to `.env` before first use. The setup script does this
automatically. All variables use the `HTSP_` prefix.

Settings are resolved in this order: **environment variable > `.env` file > default**.

| Variable                        | Type    | Default                                   | Description                                                  |
|---------------------------------|---------|-------------------------------------------|--------------------------------------------------------------|
| `HTSP_QUANTUM_BACKEND`          | string  | `simulated_annealing`                     | Solver backend: `simulated_annealing`, `cirq`, or `cudaq`    |
| `HTSP_OUTPUT_DIR`               | path    | `output`                                  | Root directory for experiment output                         |
| `HTSP_INPUT_DIR`                | path    | `input`                                   | Root directory for input datasets                            |
| `HTSP_INSTANCE_CONFIG`          | path    | `src/instance_gen_process/config.yaml`    | Path to instance generation config                           |
| `HTSP_ENABLE_NOISE_SIMULATION`  | bool    | `false`                                   | Noise kill-switch (overrides solver_config.yaml if `false`)  |
| `HTSP_RANDOM_SEED`              | int     | `42`                                      | Global random seed                                           |
| `HTSP_CUDAQ_MAX_PARALLEL_INSTANCES` | int | `1` (implicit)                          | Overrides `cudaq_max_parallel_instances` in solver YAML when set (non-empty string) |
| `HTSP_CPU_MAX_PARALLEL_INSTANCES`   | int | `1` (implicit)                          | Overrides `cpu_max_parallel_instances` in solver YAML when set (non-empty string)   |

Boolean values accept: `1`, `true`, `yes`, `on` (case-insensitive) for true;
everything else is false.

Relative paths are resolved against the project root directory.

### Noise kill-switch

When `HTSP_ENABLE_NOISE_SIMULATION=false` (the default), noise is forcibly
disabled even if `noise.enabled: true` is set in `solver_config.yaml`. This
allows quick toggling without editing YAML files.

---

## Instance config (`config.yaml`)

**Default location**: `src/instance_gen_process/config.yaml`

```yaml
n_cities: 5
n_precedences_range: [2, 5]
prices_range_hotels: [30., 150.]
prices_range_travels: [30., 150.]
seed: 42
```

### Fields

| Field                  | Type          | Required | Constraints                                                                 |
|------------------------|---------------|----------|-----------------------------------------------------------------------------|
| `n_cities`             | int           | yes      | >= 3 (depot + at least 2 cities)                                            |
| `n_precedences_range`  | [int, int]    | yes      | Both >= 0; upper bound <= `n_available * (n_available - 1) / 2`             |
| `prices_range_hotels`  | [float, float]| yes      | `[low, high]` with `low <= high`                                            |
| `prices_range_travels` | [float, float]| yes      | `[low, high]` with `low <= high`                                            |
| `seed`                 | int           | yes      | Master seed for random instance generation                                  |

### Key relationships

- `n_available = n_cities - 1`: number of non-depot cities.
- QUBO variables: `n_available^2` binary variables (one-hot encoding).
- TQUDO qudits: `n_available - 1` qudits of dimension `n_available`.
- Maximum possible precedences: `n_available * (n_available - 1) / 2`
  (acyclic DAG constraint).

---

## Solver config (`solver_config.yaml`)

**Default location**: `src/instance_gen_process/solver_config.yaml`

```yaml
n_instances: 1
# Optional: on-disk experiment workflow (and λ calibration for CPU backends); see below.
cudaq_max_parallel_instances: 1
cpu_max_parallel_instances: 1
solver: cudaq
formulation: tqudo_virtual
optimizer: COBYLA
restriction:
  lambda_0: 1000.0
  lambda_1: 1000.0
  lambda_2: 1000.0
qaoa_depth: 1
qaoa_max_iter: 100
qaoa_delta_t: 0.55
qaoa_optimizer_tol: 1.0e-6
qaoa_shots: 100000
qaoa_sample_shots: 100000
seed: 42
max_iterations: 1000
timeout_seconds: null
sa_t_initial: 1000.0
sa_t_final: 1.0e-6
sa_alpha: 0.995
noise:
  enabled: false
  noise_type: depolarizing
  probability: 0.01
```

### General fields

| Field              | Type           | Default  | Constraints                                      |
|--------------------|----------------|----------|--------------------------------------------------|
| `n_instances`      | int            | --       | >= 1, required                                   |
| `solver`           | string         | `cudaq`  | `brute_force`, `cudaq`, `cirq`, or `simulated_annealing` |
| `formulation`      | string         | `tqudo`  | `qubo`, `tqudo`, or `tqudo_virtual`              |
| `optimizer`        | string         | `COBYLA` | `COBYLA`, `Powell`, `L-BFGS-B`, `SLSQP`, `Nelder-Mead` |
| `seed`             | int or null    | null     | Optional random seed                             |
| `max_iterations`   | int            | `1000`   | >= 0 (SA max iterations)                         |
| `timeout_seconds`  | float or null  | null     | Optional timeout in seconds                      |

### Parallel instance solves (optional)

Used by the **on-disk experiment workflow** (`experiments.main_experiment_workflow` experiment modes) and by **`experiments.estimate_lambdas`** for CPU backends (`cirq`, `simulated_annealing`, `brute_force`). When greater than `1`, multiple instances are solved in separate processes; see `docs/development.md` for details.

| Field                          | Type | Default | Description |
|--------------------------------|------|---------|-------------|
| `cudaq_max_parallel_instances` | int  | `1`     | Max concurrent CUDA-Q instance solves (experiment workflow only). |
| `cpu_max_parallel_instances`   | int  | `1`     | Max concurrent workers for Cirq, brute force, and simulated annealing (experiment workflow and λ calibration). |

Non-empty env vars `HTSP_CUDAQ_MAX_PARALLEL_INSTANCES` and `HTSP_CPU_MAX_PARALLEL_INSTANCES` override the YAML values. The λ calibration CLI (`experiments.estimate_lambdas`) does not run CUDA-Q solves in parallel; use the main experiment workflow for multi-process CUDA-Q.

### Restriction (penalty coefficients)

| Field      | Type  | Default | Description                                        |
|------------|-------|---------|----------------------------------------------------|
| `lambda_0` | float | `100.0` | One city per timestep (QUBO only)                  |
| `lambda_1` | float | `100.0` | One timestep per city (no duplicates)              |
| `lambda_2` | float | `100.0` | Precedence violation penalty                       |

### QAOA parameters

| Field              | Type  | Default | Constraints                                                    |
|--------------------|-------|---------|----------------------------------------------------------------|
| `qaoa_depth`       | int   | `1`     | >= 1 (circuit depth `p`)                                       |
| `qaoa_max_iter`    | int   | `100`   | >= 1; for COBYLA must be >= `2 * qaoa_depth + 2`               |
| `qaoa_delta_t`     | float | `0.55`  | > 0 — TQA initial γ/β scale (`SolverRunConfig.delta_t`)          |
| `qaoa_optimizer_tol` | float | `1e-6` | > 0 — SciPy `minimize` tolerance for QAOA angles               |
| `qaoa_shots`       | int   | `500`   | >= 1 (shots per objective evaluation)                          |
| `qaoa_sample_shots`| int   | `1000`  | >= 1 (shots for final solution sampling)                       |

### Brute force solver (`solver: brute_force`)

Exact enumeration over the **full** discrete assignment space of the active
formulation (not permutations only). Use for baselines and sanity-checking
penalty weights.

| Formulation | Search space | Evaluations | Hard limits (`solvers/brute_force/limits.py`) |
|-------------|--------------|------------|-----------------------------------------------|
| `tqudo`     | All length-`n_available` sequences over `0..n_available-1` | `n_available^n_available` | `n_available ≤ 8` (max `8^8`) |
| `qubo`      | All `{0,1}^(n_available²)` bitstrings | `2^(n_available²)` | `(n_available²) ≤ 30` binary variables (max `2^30`) |

Not supported: `tqudo_virtual`.

Optional YAML keys (defaults match the maximum allowed full spaces):

| Field | Default | Meaning |
|-------|---------|---------|
| `brute_force_max_assignments_tqudo` | `8**8` | Abort if `n^n` would exceed this (raise cap only if you intentionally allow a smaller run) |
| `brute_force_max_assignments_qubo` | `2**30` | Abort if `2^n_vars` would exceed this |

COBYLA iteration budget is **not** validated when `solver: brute_force`.

### Simulated annealing parameters

| Field          | Type  | Default  | Constraints                            |
|----------------|-------|----------|----------------------------------------|
| `sa_t_initial` | float | `1000.0` | > 0                                    |
| `sa_t_final`   | float | `1e-6`   | > 0, < `sa_t_initial`                  |
| `sa_alpha`     | float | `0.995`  | Strictly between 0 and 1               |

Cooling schedule: `T *= sa_alpha` each step, from `sa_t_initial` down to
`sa_t_final`. Total steps before reaching `sa_t_final`:
`ceil(log(sa_t_final / sa_t_initial) / log(sa_alpha))`.

### Noise configuration

| Field              | Type           | Default          | Constraints                          |
|--------------------|----------------|------------------|--------------------------------------|
| `noise.enabled`    | bool           | `false`          | Master switch                        |
| `noise.noise_type` | string         | `"depolarizing"` | See supported types below            |
| `noise.probability`| float          | `0.01`           | In [0, 1]                            |
| `noise.gate_noise` | dict           | `{}`             | Per-gate overrides, e.g. `x: 0.02`  |

Supported noise types: `depolarizing`, `amplitude_damping`, `phase_damping`,
`bit_flip`, `phase_flip`.

Gate noise keys follow CUDA-Q naming: `"x"`, `"h"`, `"rx"`, `"rz"`,
`"cx"` / `"cnot"`, etc. When a gate is not listed, `probability` is the
fallback.

---

## Formulation-solver compatibility

| Formulation      | Cirq | CUDA-Q | Simulated Annealing | Brute force | Additional constraint              |
|------------------|:----:|:------:|:-------------------:|:-----------:|------------------------------------|
| `tqudo`          |  Y   |   --   |         Y           |      Y      | BF: `n_available ≤ 8`              |
| `tqudo_virtual`  |  Y   |   Y    |         --          |     --      | `n_available` must be power of two |
| `qubo`           |  Y   |   Y    |         Y           |      Y      | Quantum: ≤ 30 qubits; BF: ≤ 30 binary vars |

Invalid combinations raise `ValueError` at validation time with a descriptive
message suggesting alternatives.

### Qubit/qudit counts by formulation

| Formulation      | Quantum systems                         | Hilbert space dimension     |
|------------------|-----------------------------------------|-----------------------------|
| `tqudo`          | `n_available - 1` qudits (dim `d`)     | `d^(n_available-1)`         |
| `tqudo_virtual`  | `(n_available - 1) * ceil(log2(d))` qubits | `2^n_qubits`           |
| `qubo`           | `n_available^2` qubits                  | `2^(n_available^2)`         |

### Scaling examples

| `n_cities` | `n_available` | TQUDO qudits | TQUDO virtual qubits | QUBO qubits |
|------------|---------------|-------------:|---------------------:|------------:|
| 3          | 2             | 1            | 1                    | 4           |
| 5          | 4             | 3            | 6                    | 16          |
| 9          | 8             | 7            | 21                   | 64          |
| 17         | 16            | 15           | 60                   | 256         |

---

## Parameter tuning guidance

### Lambda penalties

- Lambdas must be large enough relative to cost terms to penalise constraint
  violations. A good starting point: set lambdas 5-10x larger than the maximum
  single-step cost.
- With `prices_range = [30, 150]` and `n_cities = 5`, setting
  `lambda_0 = lambda_1 = lambda_2 = 1000` works well.
- `lambda_0` only affects QUBO. Setting it too low causes the QUBO solver to
  find infeasible one-hot violations.
- `lambda_2` controls precedence enforcement. If solutions frequently violate
  precedences, increase it.

### Energy normalisation

Both formulations normalise cost tensors/matrices so that
`max(abs(values)) == 1`. This is important for QAOA because gate angles scale
with the energy values. The `energy_scale` factor is stored in the problem
object and automatically applied when computing costs.

### QAOA depth

- `qaoa_depth = 1` is the simplest ansatz; sufficient for small instances.
- Higher depth increases expressiveness but also the number of optimisation
  parameters (`2 * depth`) and circuit complexity.
- COBYLA budget must satisfy `qaoa_max_iter >= 2 * qaoa_depth + 2`.

### Shot counts

- `qaoa_shots`: more shots per cost evaluation give more accurate gradient
  estimates but increase runtime linearly.
- `qaoa_sample_shots`: more shots for final sampling increase the probability
  of observing the optimal solution.
- For noisy simulations, higher shot counts help average out noise.

### SA parameters

- `sa_alpha` close to 1 (e.g. 0.999) gives slower cooling and better
  exploration but longer runtime.
- `sa_t_initial` should be high enough that most uphill moves are accepted
  initially.
- `sa_t_final` should be low enough that only downhill moves are accepted
  at convergence.
