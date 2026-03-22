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
- `solver` in `{cudaq, cirq, simulated_annealing}`.
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
- `tqudo_virtual` on SA (unsupported).
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
    noise_config: NoiseConfig = NoiseConfig()  # Noise simulation settings
    sa_t_initial: float = 1000.0             # SA initial temperature
    sa_t_final: float = 1e-6                 # SA final temperature
    sa_alpha: float = 0.995                  # SA geometric cooling factor
```

#### SolverResult

```python
@dataclass(frozen=True, slots=True)
class SolverResult:
    solver_name: str           # e.g. "cirq", "cudaq", "simulated_annealing"
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

## utils

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

#### run_workflow

```python
def run_workflow(
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
    settings: Settings | None = None,
) -> None
```

Full pipeline: load configs, generate instances, validate, solve each, save
JSON results incrementally. Handles SIGINT gracefully (finishes current
instance). Applies environment noise kill-switch when `settings` is provided.

#### main

```python
def main() -> None
```

CLI entry point with `--instance-config`, `--solver-config`, and `--output`
arguments. Loads `Settings` from `.env` and passes them to `run_workflow()`.
