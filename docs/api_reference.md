# API Reference

Public API for all modules in the Hotels-TSP-Tensor-QUDO project.
Package root is `src/`, so imports use bare module names
(e.g. `from solvers.base import SolverRunConfig`).

---

## instance_gen_process

### Models (`instance_gen_process/models.py`)

#### InstanceConfig

Configuration for random instance generation.

```python
@dataclass(frozen=True, slots=True)
class InstanceConfig:
    n_cities: int                              # Total cities including depot (>= 3)
    n_precedences_range: tuple[int, int]       # [low, high] range for precedence count
    prices_range_hotels: tuple[float, float]   # [low, high] uniform range for hotel prices
    prices_range_travels: tuple[float, float]  # [low, high] uniform range for travel prices
    seed: int = 42                             # Master seed for reproducibility
```

#### ProblemInstance

Canonical in-memory problem representation consumed by all solvers.

```python
@dataclass(frozen=True, slots=True)
class ProblemInstance:
    n_cities: int                                  # Total cities including depot
    precedences: tuple[tuple[int, int], ...]       # (origin, destination) precedence pairs
    prices_hotels: np.ndarray                      # shape (n_available, n_available)
    prices_travels: np.ndarray                     # shape (n_cities, n_cities, n_cities)
    seed: int = 0                                  # Seed used to generate this instance
```

- `n_available = n_cities - 1` (depot excluded from decision variables).
- `prices_hotels[t, a]`: cost of staying in city `a` at timestep `t`.
- `prices_travels[t, a, b]`: cost of traveling from `a` to `b` at timestep `t`.
- `prices_travels[:, i, i] = 0` (self-loops have zero cost).

#### ProblemTQUDO

Tensor-QUDO formulation for quantum backends.

```python
@dataclass(frozen=True, slots=True)
class ProblemTQUDO:
    Etab: np.ndarray          # 3D tensor, shape (n_available, d, d)
    Ettprimeab: np.ndarray    # 4D tensor, shape (n_available, n_available, d, d)
    energy_scale: float = 1.0 # Normalisation factor
```

- `d = n_available` (qudit dimension).
- Tensors are normalised so `max(|Etab|, |Ettprimeab|) == 1`.
- Multiply sampled costs by `energy_scale` to recover original units.

#### ProblemQUBO

QUBO formulation for quantum/classical backends.

```python
@dataclass(frozen=True, slots=True)
class ProblemQUBO:
    qubo_matrix: np.ndarray   # Symmetric, shape (n_vars, n_vars), n_vars = n_available^2
    energy_scale: float = 1.0 # Normalisation factor
```

- Normalised so all entries lie in `[-1, 1]`.
- `QUBO_cost = real_cost - (lambda_0 + lambda_1) * n_available` for feasible solutions.

#### RestrictionConfig

Penalty coefficients for constraint encoding.

```python
@dataclass(frozen=True, slots=True)
class RestrictionConfig:
    lambda_0: float  # One city per timestep (QUBO only)
    lambda_1: float  # One timestep per city (no duplicates)
    lambda_2: float  # Precedence violation penalty
```

- TQUDO uses only `lambda_1` and `lambda_2` (qudit encoding handles `lambda_0`).
- QUBO requires all three.

---

### Generator (`instance_gen_process/generator.py`)

#### generate_random_set_instances

```python
def generate_random_set_instances(
    config: InstanceConfig,
    n_instances: int,
    seed: int = 42,
) -> list[ProblemInstance]
```

Generates `n_instances` random instances from a master RNG seeded with `seed`.
Each instance receives a unique sub-seed stored in `ProblemInstance.seed` for
individual reproducibility.

#### generate_random_instance

```python
def generate_random_instance(
    config: InstanceConfig,
    seed: int,
) -> ProblemInstance
```

Generates a single instance with deterministic randomness from `seed`. Samples
precedences (rejecting cycles via `would_create_cycle()`), hotel prices, and
travel prices from the configured ranges. Logs a warning if the requested
number of precedences cannot be achieved due to acyclicity constraints.

#### generate_TQUDO_from_problem

```python
def generate_TQUDO_from_problem(
    problem: ProblemInstance,
    restriction: RestrictionConfig,
) -> ProblemTQUDO
```

Builds `Etab` (travel + hotel costs including closed-loop legs) and
`Ettprimeab` (`lambda_1` duplicate penalties + `lambda_2` precedence
penalties). Normalises both tensors by `energy_scale`.

#### generate_QUBO_from_problem

```python
def generate_QUBO_from_problem(
    problem: ProblemInstance,
    restriction: RestrictionConfig,
) -> ProblemQUBO
```

Builds a symmetric `qubo_matrix` encoding costs and all three penalty terms.
Uses linear indexing `idx(t, i) = t * n_available + i`. Normalises by
`energy_scale`.

---

### Config loaders

#### load_instance_config (`config_loader.py`)

```python
def load_instance_config(path: Path | str | None = None) -> InstanceConfig
```

Loads and validates `config.yaml`. Default path:
`src/instance_gen_process/config.yaml`.

Validation rules:
- `n_cities >= 3` (depot + at least 2 available cities).
- `n_precedences_range`: both values non-negative,
  upper bound <= `n_available * (n_available - 1) // 2`.
- `prices_range_hotels`, `prices_range_travels`: two-element `[low, high]`
  with `low <= high`.
- `seed`: required integer.

#### load_solver_config (`solver_config_loader.py`)

```python
def load_solver_config(path: Path | str | None = None) -> dict[str, Any]
```

Loads and validates `solver_config.yaml`. Returns a dict with all solver
parameters. Validates:
- `n_instances >= 1`.
- `solver` in `{brute_force, cudaq, cirq, simulated_annealing}`.
- `formulation` in `{qubo, tqudo, tqudo_virtual}`.
- `optimizer` in `{COBYLA, Powell, L-BFGS-B, SLSQP, Nelder-Mead}`.
- COBYLA budget: `qaoa_max_iter >= 2 * qaoa_depth + 2`.
- SA parameters: `sa_t_initial > 0`, `sa_t_final > 0`,
  `sa_t_final < sa_t_initial`, `0 < sa_alpha < 1`.
- Noise config: validated `NoiseConfig` from the `noise:` block.

#### solver_config_to_run_config

```python
def solver_config_to_run_config(config: dict[str, Any]) -> SolverRunConfig
```

Converts the loaded solver config dict into a `SolverRunConfig` dataclass.

#### validate_solver_instance_compatibility

```python
def validate_solver_instance_compatibility(
    instance_config: InstanceConfig,
    solver_config: dict[str, Any],
) -> None
```

Cross-validates instance and solver configurations. Raises `ValueError` for:
- QUBO on quantum backends with > 30 qubits.
- Native `tqudo` on CUDA-Q (unsupported).
- `tqudo_virtual` on SA or brute force (unsupported).
- `brute_force` with formulation other than `qubo` or `tqudo`, or instance size
  above brute-force limits / optional assignment caps.
- `tqudo_virtual` when `n_available` is not a power of two.

---

## solvers

### Base types (`solvers/base.py`)

#### OptimizerType

```python
OptimizerType = Literal["COBYLA", "Powell", "L-BFGS-B", "SLSQP", "Nelder-Mead"]
```

#### SolverRunConfig

```python
@dataclass(frozen=True, slots=True)
class SolverRunConfig:
    max_iterations: int = 1_000              # SA max iterations
    timeout_seconds: float | None = None     # Optional timeout
    formulation: str = "tqudo"               # "qubo" | "tqudo" | "tqudo_virtual"
    restriction_config: RestrictionConfig | None = None  # Penalty coefficients
    qaoa_depth: int = 1                      # QAOA circuit depth (p)
    qaoa_max_iter: int = 100                 # Optimizer max iterations
    qaoa_shots: int = 500                    # Shots per cost evaluation
    qaoa_sample_shots: int = 1000            # Shots for final solution sampling
    seed: int | None = None                  # Random seed
    optimizer: OptimizerType = "COBYLA"      # scipy optimizer method
    delta_t: float = 0.55                    # TQA parameter scheduling scale
    optimizer_tol: float = 1e-6              # QAOA classical optimizer tolerance
    noise_config: NoiseConfig = NoiseConfig()  # Noise simulation settings
    sa_t_initial: float = 1000.0             # SA initial temperature
    sa_t_final: float = 1e-6                 # SA final temperature
    sa_alpha: float = 0.995                  # SA geometric cooling factor
    brute_force_max_assignments_tqudo: int = 8**8   # brute_force: max n^n
    brute_force_max_assignments_qubo: int = 2**30   # brute_force: max 2^n_vars
```

#### SolverResult

```python
@dataclass(frozen=True, slots=True)
class SolverResult:
    solver_name: str           # e.g. "cirq", "cudaq", "simulated_annealing", "brute_force"
    objective_value: float     # Raw objective in original problem units
    feasible: bool             # Whether solution satisfies all constraints
    runtime_seconds: float     # Wall-clock time for solve()
    metadata: dict[str, Any]   # Best sequence, energy history, angles, samples, real_cost
```

Metadata keys (when available):
- `best_sequence`: list of city indices.
- `best_bitstring`: measurement outcome string.
- `best_binary`: binary solution vector (QUBO only).
- `real_cost`: formulation-independent cost (feasible solutions only).
- `initial_energy`, `energy_history`: optimisation trace.
- `optimal_angles`: `{"gamma": [...], "beta": [...]}`.
- `initial_samples`, `final_samples`: `{bitstring: count}` dicts.

#### SolverProtocol

```python
class SolverProtocol(Protocol):
    solver_name: str
    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult: ...
```

---

### Noise (`solvers/noise.py`)

#### NoiseConfig

```python
@dataclass(frozen=True, slots=True)
class NoiseConfig:
    enabled: bool = False
    noise_type: NoiseModelType = "depolarizing"
    probability: float = 0.01           # Error probability in [0, 1]
    gate_noise: dict[str, float] = {}   # Per-gate overrides, e.g. {"rx": 0.02}
```

Validates `noise_type` against `VALID_NOISE_TYPES` and all probabilities in
`[0, 1]` at construction time via `__post_init__`.

#### NoiseModelType

```python
NoiseModelType = Literal[
    "depolarizing", "amplitude_damping", "phase_damping", "bit_flip", "phase_flip"
]
```

#### warn_if_large_system

```python
def warn_if_large_system(
    self,
    n_qubits: int,
    *,
    gpu_trajectory: bool = False,
    qudit_dimension: int = 2,
) -> None
```

Logs memory scaling warnings when `n_qubits * log2(qudit_dimension) > 15`.

---

### BaseQAOASolver (`solvers/_qaoa_base.py`)

Abstract base class shared by `CirqSolver` and `CudaqSolver`.

```python
class BaseQAOASolver(ABC):
    solver_name: str

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult: ...

    @abstractmethod
    def _get_tqudo_runner(self) -> Callable[..., dict] | None: ...
    @abstractmethod
    def _get_tqudo_virtual_runner(self) -> Callable[..., dict] | None: ...
    @abstractmethod
    def _get_qubo_runner(self) -> Callable[..., dict] | None: ...
    @abstractmethod
    def _serialize_samples(self, samples: Any) -> dict[str, int] | None: ...
    @abstractmethod
    def _noise_qubit_count(self, instance, formulation) -> tuple[int, dict[str, Any]]: ...
```

- Returning `None` from a runner signals that the formulation is not supported.
- `solve()` dispatches to `_solve_tqudo()` or `_solve_qubo()`, handles timing,
  feasibility validation, and metadata construction.

---

### Brute force (`solvers/brute_force/solver.py`)

#### BruteForceSolver

```python
class BruteForceSolver:
    solver_name: str  # "brute_force"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult: ...
```

Exhaustive search over the full QUBO or TQUDO assignment space (not tour
permutations only). Raises `ValueError` for unsupported formulations or sizes
outside `solvers/brute_force/limits.py` / `run_config` caps.

---

## utils

The `utils` package re-exports common helpers from `utils/__init__.py` using
**lazy** loading (`__getattr__`), so `from utils.output_paths import …` (and
similar submodule imports) do not eagerly import `utils.constraints` and
avoid circular-import issues with `instance_gen_process` and `data_analysis`.

### Native stderr (`utils/native_stderr.py`)

Helpers to redirect OS file descriptor 2 during CUDA-Q runs so native library
stderr does not corrupt single-line TTY progress. Controlled by
`HTSP_SILENCE_NATIVE_STDERR` and optional `HTSP_NATIVE_STDERR_LOG`; see
`docs/configuration.md`.

#### silence_native_stderr_requested

```python
def silence_native_stderr_requested() -> bool
```

Returns whether `HTSP_SILENCE_NATIVE_STDERR` is set to a truthy token (`1`, `true`,
`yes`, `on`). Falsy/unset disables redirection.

#### resolve_native_stderr_log_path

```python
def resolve_native_stderr_log_path(output_root: Path) -> Path
```

Returns `HTSP_NATIVE_STDERR_LOG` if set, otherwise `output_root / "native_stderr.log"`
(resolved).

#### redirect_native_stderr_to_file

```python
@contextmanager
def redirect_native_stderr_to_file(log_path: Path) -> Generator[None, None, None]
```

For the duration of the context, duplicates fd 2 to *log_path* (append), redirects
`sys.stderr`, and restores both on exit.

---

### Costs (`utils/costs.py`)

#### calculate_qubo_cost

```python
def calculate_qubo_cost(problem: ProblemQUBO, solution: np.ndarray) -> float
```

Returns `x^T Q x * energy_scale`. Input `solution` can be shape `(n_vars,)`
or `(n_vars, 1)`.

#### calculate_tqudo_cost

```python
def calculate_tqudo_cost(problem: ProblemTQUDO, solution: np.ndarray) -> float
```

Returns `(sum Etab + sum Ettprimeab) * energy_scale`. Uses vectorised NumPy
indexing for efficient computation.

#### calculate_real_cost

```python
def calculate_real_cost(problem: ProblemInstance, sequence: list[int]) -> float
```

Returns `hotel_cost + travel_cost` in original units. The sequence must have
length `n_available`. Travel includes depot-to-first, inter-city segments, and
last-to-depot legs.

Raises `ValueError` if `len(sequence) != n_available`.

---

### Batch costs (`utils/costs_batch.py`)

Vectorised objective evaluation (same algebra as `calculate_qubo_cost` /
`calculate_tqudo_cost`). Used by brute-force enumeration and available for
other batch tooling.

#### unpack_qubo_bitmatrix / batch_qubo_costs

Decode integer indices to bit rows `(B, n_vars)` and compute per-row
`x @ Q @ x * energy_scale`.

#### unpack_tqudo_sequences / batch_tqudo_costs

Mixed-radix expansion of assignment indices to city sequences `(B, n_available)`
and batched TQUDO energy from `Etab` and `Ettprimeab`.

---

### Constraints (`utils/constraints.py`)

#### idx

```python
def idx(t: int, i: int, n_available: int) -> int
```

Linear index for QUBO binary vector: `t * n_available + i`.

#### validate_instance_constraints

```python
def validate_instance_constraints(instance: ProblemInstance) -> bool
```

Checks: `n_available >= 1`, array shapes, precedence bounds within range,
no self-loops, no cycles in precedence graph.

#### would_create_cycle

```python
def would_create_cycle(
    precedences: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: int,
    destination: int,
) -> bool
```

BFS reachability check: returns `True` if adding `(origin, destination)` would
create a cycle.

#### validate_solution_constraints_tqudo

```python
def validate_solution_constraints_tqudo(
    instance: ProblemInstance,
    solution: list[int] | np.ndarray,
) -> bool
```

Checks: length equals `n_available`, no duplicates, all indices in
`[0, n_available)`, precedences satisfied.

#### validate_solution_constraints_qubo

```python
def validate_solution_constraints_qubo(
    instance: ProblemInstance,
    solution: np.ndarray,
) -> bool
```

Decodes binary vector via `qubo_binary_to_sequence()`, then validates
precedence and uniqueness.

#### qubo_binary_to_sequence

```python
def qubo_binary_to_sequence(solution: np.ndarray, n_available: int) -> np.ndarray | None
```

Decodes one-hot binary vector to city sequence. Returns `None` if encoding is
invalid (not exactly one active bit per timestep, or duplicate cities).

#### sequence_to_qubo_binary

```python
def sequence_to_qubo_binary(
    sequence: np.ndarray | list[int],
    n_available: int,
) -> np.ndarray
```

Encodes city sequence as one-hot binary vector of shape `(n_available^2,)`.

---

### JSON (`utils/json_serialize.py`)

#### to_json_friendly

```python
def to_json_friendly(obj: Any) -> Any
```

Recursively normalises values for JSON: non-finite floats → `None`, lists,
dicts, NumPy `.tolist()`, dataclasses via `dataclasses.asdict()`.

---

### Experiment snapshots (`utils/experiment_serialize.py`)

Serialisers for CLI outputs shared by `experiments/main_experiment_workflow.py`,
`experiments/cudaq_parallel.py`, `estimate_lambdas.py`, and `estimate_t0.py`.

- `serialize_instance_config(config: InstanceConfig) -> dict[str, Any]`
- `serialize_restriction_config(restriction: RestrictionConfig) -> dict[str, float]`
- `serialize_solver_result(result: SolverResult) -> dict[str, Any]`
- `solver_config_payload_dict(solver_config_dict: dict[str, Any]) -> dict[str, Any]` —
  expands the loaded YAML’s `restriction` dataclass and runs `to_json_friendly`.

---

### YAML (`utils/yaml_tools.py`)

- `load_yaml_mapping(path: Path | str) -> dict[str, Any]` — safe load; empty file → `{}`.
- `merge_solver_yaml_dicts(base, override) -> dict[str, Any]` — deep merge with
  nested `restriction` and `noise` dict merging.

---

### Experiment disk paths (`utils/experiment_paths.py`)

Path helpers for the on-disk workflow layout:

- `instances_raw_dir(output_root, n_cities)`
- `solutions_solver_root(output_root, solver)`
- `solutions_raw_dir(output_root, solver, formulation, n_cities, qaoa_depth)`
- `instance_json_path(output_root, n_cities, index_one_based)`

`experiments/workflow_io.py` imports these (and YAML helpers) and adds instance
JSON load/save, `load_instance_generation_entries`, `experiment_depth_iterations`,
etc.

---

### QAOA helpers (`utils/qaoa_helpers.py`)

#### tqa_init_params

```python
def tqa_init_params(depth: int, delta_t: float) -> np.ndarray
```

Returns TQA initial parameters as a 1D array of shape `(2 * depth,)`:
`[gamma_1..gamma_p, beta_1..beta_p]` where `gamma_i = (i/p) * delta_t`
and `beta_i = (1 - i/p) * delta_t`.

#### bitstring_to_binary

```python
def bitstring_to_binary(bitstring: str) -> np.ndarray
```

Converts `"01101"` to `np.array([0, 1, 1, 0, 1])`.

#### most_probable_key

```python
def most_probable_key(counts: dict[str, int], fallback: str) -> str
```

Returns the key with the highest count. Returns `fallback` if `counts` is
empty.

#### measurement_histogram_for_json

```python
def measurement_histogram_for_json(
    samples: Mapping[str, Any] | None,
) -> dict[str, int] | None
```

Normalises backend shot histograms (Cirq, CUDA-Q) to string keys, integer
counts, sorted by descending count. Returns `None` when `samples` is `None`.

#### is_power_of_two

```python
def is_power_of_two(value: int) -> bool
```

Returns `True` when `value > 0 and (value & (value - 1)) == 0`.

---

### Optimizer (`utils/optimizer.py`)

#### minimize_options

```python
def minimize_options(method: str, max_iter: int) -> dict
```

Builds `scipy.optimize.minimize` options dict. Handles method-specific option
names: `maxfev` for Nelder-Mead/Powell, `maxfun` for L-BFGS-B.

---

### Output paths (`utils/output_paths.py`)

#### OutputLayout

```python
@dataclass(frozen=True, slots=True)
class OutputLayout:
    root: Path
    raw: Path
    processed: Path
    images: Path
```

#### build_output_layout

```python
def build_output_layout(root: Path) -> OutputLayout
```

Returns `OutputLayout(root, root/"raw", root/"processed", root/"images")`.
Does not create directories.

---

### Progress (`utils/progress.py`)

#### ProgressReporter

Singleton-based progress display for experiment runs.

```python
class ProgressReporter:
    def configure(self, n_instances: int) -> None: ...
    def instance_start(self, i: int) -> None: ...
    def opt_step(self, step: int, max_steps: int, energy: float) -> None: ...
    def instance_done(self, i: int, path: str) -> None: ...
```

- `opt_step()` overwrites a single line with `\r`, showing a progress bar
  and current energy. Checkpoints every `max_steps // 10` steps.
- Module-level singleton: `reporter = ProgressReporter()`.

---

### Logging (`utils/logging_utils.py`)

#### configure_logging

```python
def configure_logging(level: str = "INFO") -> None
```

Configures project-wide logging with format:
`%(asctime)s | %(levelname)s | %(name)s | %(message)s`.

---

## math_utils

### QUBO to Ising (`math_utils/qubo_ising.py`)

#### qubo_to_ising

```python
def qubo_to_ising(qubo_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]
```

Converts a symmetric QUBO matrix to Ising form via the substitution
`x_i = (1 - s_i) / 2`:

- `h[i] = -0.5 * sum_j Q[i, j]` -- local fields.
- `J[i, j] = Q[i, j] / 2` for `i < j` -- upper-triangular couplings.
- `offset = trace(Q)/2 + sum_{i<j} Q[i,j] / 2` -- constant energy shift.

Raises `ValueError` if `qubo_matrix` is not square or not symmetric.

---

## config

### Settings (`config/settings.py`)

#### Settings

```python
@dataclass(frozen=True, slots=True)
class Settings:
    quantum_backend: BackendName    # "simulated_annealing" | "cirq" | "cudaq"
    output_dir: Path                # Resolved output directory
    input_dir: Path                 # Resolved input directory
    instance_config_path: Path      # Resolved path to config.yaml
    enable_noise_simulation: bool   # Noise simulation kill-switch
    random_seed: int                # Global random seed
```

#### load_settings

```python
def load_settings(
    env_file: Path | str | None = None,
    project_root: Path | None = None,
) -> Settings
```

Resolution order for each setting: environment variable > `.env` file > default.
All `HTSP_*` prefixed variables. Relative paths are resolved against
`project_root` (which defaults to the repository root).

Raises `ValueError` for unsupported backend names or non-integer seed values.

---

## experiments

### Workflow (`experiments/main_experiment_workflow.py`)

#### run_experiment_from_yaml

```python
def run_experiment_from_yaml(
    experiment_yaml_path: Path | str,
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
    settings: Settings | None = None,
) -> None
```

Merges experiment YAML with `solver_config.yaml`, loads instances from disk
(`raw/instances/`), solves, writes `raw/solutions/...`. Applies noise kill-switch
when `settings` is provided.

#### run_experiment_batch

```python
def run_experiment_batch(
    experiment_yaml_paths: list[Path],
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
    settings: Settings | None = None,
) -> None
```

Runs `run_experiment_from_yaml` for each path in order.

#### run_generate_instances

```python
def run_generate_instances(
    instance_config_path: Path | str | None = None,
    instance_generation_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
) -> None
```

Writes `raw/instances/n_<n_cities>/instance_<k>.json` from the generation config.

#### main

```python
def main() -> None
```

CLI entry point with required `--mode`, plus `--instance-config`, `--solver-config`,
`--output`, `--experiment-yaml`, `--check-solver`, and related flags. Loads
`Settings` from `.env` and passes them to experiment runs where applicable.

### Calibration (`experiments/estimate_t0.py`, `experiments/estimate_lambdas.py`)

- **`estimate_t0`**: `run_estimation(...)` — samples random instances, runs Ben–Ameur
  `estimate_initial_temperature`, prints a recommended median `T₀`, writes JSON
  under `--output` (default `output/T0sampling`).
- **`estimate_lambdas`**: `run_lambda_sampling(...)` — grid search over λ triples
  (`--lambda-values`), ranks by feasibility and mean real cost (heuristic solvers)
  or mean gap to combinatorial optimum (`brute_force`). Merges parallel-instance
  YAML keys (`cpu_max_parallel_instances`, `cudaq_max_parallel_instances`) from the
  solver file for `resolve_cpu_max_parallel_instances` (CUDA-Q stays sequential).

### Workflow I/O (`experiments/workflow_io.py`)

YAML merge, instance JSON round-trip, and re-exports of
`utils.yaml_tools` / `utils.experiment_paths` helpers for experiment CLIs and
tests. See **YAML** and **Experiment disk paths** under `utils` above.

---

## data_analysis

Post-processing package (optional `analysis` extra: pandas, pyarrow, matplotlib).
Reads JSON under `output/raw/` and writes tables to `output/processed/` and
figures to `output/images/`.

### Package entry (`data_analysis/__init__.py`)

Lazy exports: `process_raw_results`, `run_pipeline` (from `data_analysis.pipeline`).

### Pipeline (`data_analysis/pipeline.py`)

- `run_pipeline(output_root, manifest_format=..., sample_quality=..., skip_plots=..., energy_curve_percentiles=...)` — keyword-only `energy_curve_percentiles` defaults to `True`.
- `process_raw_results(raw_dir, processed_dir)` — requires `processed_dir.name == "processed"` and `processed_dir.parent == raw_dir.parent` (same output root).

CLI: `python -m data_analysis.pipeline --output-root output [--format parquet|csv] [--sample-quality] [--skip-plots] [--no-energy-curve-percentiles]`

### Ingest (`data_analysis/ingest.py`)

CLI: `python -m data_analysis.ingest --output-root output` — builds
`processed/manifest.parquet` or `.csv` from JSON under `raw/solutions/**/*.json`.

Manifest rows (`data_analysis/records.py`): `parse_ok` means the file is valid JSON
with a top-level object; `solve_ok` means `solver_output` is a normal result (no
`error` key). Failed solves stored by the workflow have `parse_ok` True and
`solve_ok` False. Metrics aggregate successful runs using `parse_ok & solve_ok`;
older manifests without `solve_ok` infer it from a null `solver_error`.

### Metrics (`data_analysis/metrics.py`)

CLI: `python -m data_analysis.metrics --output-root output [--sample-quality] [--no-energy-curve-percentiles]` —
paired metrics, summaries, and energy-curve aggregates.

### Plots (`data_analysis/plot.py`)

CLI: `python -m data_analysis.plot --output-root output` — PNGs under `images/` (subfolders: `energy_history/`, `dashboards/`, `approx_ratio/`, `steps/`, `improvement/`, `p_opt/`).

Supporting modules: `scan.py`, `records.py`; `benchmark/run.py` (dashboards, \(\rho\), \(P(\mathrm{opt})\), energy / \(\Delta P\) comparisons — some series re-read JSON for histograms and `energy_history`); `optimal_sample_mass.py` (brute-force optimal sequence → histogram keys, sample mass); `energy_plots.py`; layout via `utils.output_paths`.
