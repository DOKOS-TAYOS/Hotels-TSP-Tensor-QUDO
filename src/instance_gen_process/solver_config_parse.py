"""Parse and validate solver configuration from YAML mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from instance_gen_process.models import RestrictionConfig
from utils.yaml_tools import read_solver_yaml_as_mapping

DEFAULT_SOLVER_CONFIG_PATH = Path(__file__).with_name("solver_config.yaml")

VALID_SOLVERS = frozenset({"brute_force", "cudaq", "cirq", "simulated_annealing"})
VALID_FORMULATIONS = frozenset({"qubo", "tqudo", "tqudo_virtual"})
VALID_OPTIMIZERS = frozenset({"COBYLA", "Powell", "L-BFGS-B", "SLSQP", "Nelder-Mead"})


def _parse_int_setting(raw_value: Any, field_name: str, minimum: int) -> int:
    """Parse an integer YAML field and enforce *minimum*.

    Args:
        raw_value: Raw scalar from YAML.
        field_name: Key name for error messages.
        minimum: Inclusive lower bound (or strictness per field semantics).

    Returns:
        Parsed integer ``>= minimum`` when minimum is non-negative.

    Raises:
        ValueError: If the value is below *minimum*.

    """
    value = int(raw_value)
    if value < minimum:
        comparator = "at least" if minimum > 0 else "greater than or equal to"
        raise ValueError(f"{field_name} must be {comparator} {minimum}")
    return value


def _validate_cobyla_budget(qaoa_depth: int, qaoa_max_iter: int, optimizer: str) -> None:
    """Ensure COBYLA's iteration budget covers the number of QAOA angles.

    Args:
        qaoa_depth: QAOA layers p (2p variational parameters).
        qaoa_max_iter: Requested optimizer iterations.
        optimizer: Optimizer name; no-op unless ``COBYLA``.

    Raises:
        ValueError: If COBYLA is selected and ``qaoa_max_iter`` is too small.

    """
    if optimizer != "COBYLA":
        return

    minimum_iterations = (2 * qaoa_depth) + 2
    if qaoa_max_iter < minimum_iterations:
        raise ValueError(
            "qaoa_max_iter is too small for COBYLA. "
            f"With qaoa_depth={qaoa_depth}, it must be at least {minimum_iterations}."
        )


def parse_solver_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a solver configuration mapping.

    Same rules as :func:`load_solver_config` but accepts an already-loaded dict
    (e.g. after merging experiment YAML with a base file). ``qaoa_depth`` must
    be a scalar integer in *data*.

    Args:
        data: Raw mapping (typically from YAML).

    Returns:
        Normalised dict suitable for :func:`~instance_gen_process.solver_run_config_map.solver_config_to_run_config`.

    Raises:
        ValueError: If required fields are missing or invalid.

    """
    if "n_instances" not in data:
        raise ValueError("Missing required field: n_instances")
    n_instances = int(data["n_instances"])
    if n_instances < 1:
        raise ValueError("n_instances must be at least 1")

    solver = data.get("solver", "cudaq")
    if solver not in VALID_SOLVERS:
        raise ValueError(f"solver must be one of {sorted(VALID_SOLVERS)}, got: {solver!r}")

    formulation = data.get("formulation", "tqudo")
    if formulation not in VALID_FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {sorted(VALID_FORMULATIONS)}, got: {formulation!r}"
        )

    optimizer = data.get("optimizer", "COBYLA")
    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}, got: {optimizer!r}"
        )

    restriction_data = data.get("restriction") or {}
    restriction = RestrictionConfig(
        lambda_0=float(restriction_data.get("lambda_0", 100.0)),
        lambda_1=float(restriction_data.get("lambda_1", 100.0)),
        lambda_2=float(restriction_data.get("lambda_2", 100.0)),
    )

    qaoa_depth = _parse_int_setting(data.get("qaoa_depth", 1), "qaoa_depth", minimum=1)
    qaoa_max_iter = _parse_int_setting(
        data.get("qaoa_max_iter", 100), "qaoa_max_iter", minimum=1
    )
    qaoa_delta_t = float(data.get("qaoa_delta_t", 0.55))
    if qaoa_delta_t <= 0:
        raise ValueError("qaoa_delta_t must be positive")
    qaoa_optimizer_tol = float(data.get("qaoa_optimizer_tol", 1e-6))
    if qaoa_optimizer_tol <= 0:
        raise ValueError("qaoa_optimizer_tol must be positive")
    qaoa_shots = _parse_int_setting(data.get("qaoa_shots", 500), "qaoa_shots", minimum=1)
    qaoa_sample_shots = _parse_int_setting(
        data.get("qaoa_sample_shots", 1000), "qaoa_sample_shots", minimum=1
    )
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)
    max_iterations = _parse_int_setting(
        data.get("max_iterations", 1000), "max_iterations", minimum=0
    )
    timeout_seconds = data.get("timeout_seconds")
    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)

    sa_t_initial = float(data.get("sa_t_initial", 1000.0))
    if sa_t_initial <= 0:
        raise ValueError("sa_t_initial must be positive")
    sa_t_final = float(data.get("sa_t_final", 1e-6))
    if sa_t_final <= 0:
        raise ValueError("sa_t_final must be positive")
    if sa_t_final >= sa_t_initial:
        raise ValueError("sa_t_final must be less than sa_t_initial")
    sa_alpha = float(data.get("sa_alpha", 0.995))
    if not (0 < sa_alpha < 1):
        raise ValueError("sa_alpha must be between 0 and 1 (exclusive)")

    from solvers.brute_force.limits import QUBO_MAX_BINARY_VARS, TQUDO_MAX_N_AVAILABLE

    brute_force_max_assignments_tqudo = _parse_int_setting(
        data.get(
            "brute_force_max_assignments_tqudo",
            TQUDO_MAX_N_AVAILABLE**TQUDO_MAX_N_AVAILABLE,
        ),
        "brute_force_max_assignments_tqudo",
        minimum=1,
    )
    brute_force_max_assignments_qubo = _parse_int_setting(
        data.get("brute_force_max_assignments_qubo", 2**QUBO_MAX_BINARY_VARS),
        "brute_force_max_assignments_qubo",
        minimum=1,
    )

    if solver != "brute_force":
        _validate_cobyla_budget(qaoa_depth, qaoa_max_iter, optimizer)

    from solvers.noise import NoiseConfig, NoiseModelType, VALID_NOISE_TYPES

    noise_data = data.get("noise") or {}
    noise_enabled = bool(noise_data.get("enabled", False))
    noise_type_raw = noise_data.get("noise_type", "depolarizing")
    if noise_type_raw not in VALID_NOISE_TYPES:
        raise ValueError(
            f"noise.noise_type must be one of {sorted(VALID_NOISE_TYPES)}, "
            f"got: {noise_type_raw!r}"
        )
    noise_type: NoiseModelType = noise_type_raw  # validated above
    noise_probability = float(noise_data.get("probability", 0.01))
    if not 0.0 <= noise_probability <= 1.0:
        raise ValueError(f"noise.probability must be in [0, 1], got {noise_probability}")
    gate_noise_raw = noise_data.get("gate_noise") or {}
    gate_noise = {str(k): float(v) for k, v in gate_noise_raw.items()}
    noise_config = NoiseConfig(
        enabled=noise_enabled,
        noise_type=noise_type,
        probability=noise_probability,
        gate_noise=gate_noise,
    )

    return {
        "n_instances": n_instances,
        "solver": solver,
        "formulation": formulation,
        "optimizer": optimizer,
        "restriction": restriction,
        "qaoa_depth": qaoa_depth,
        "qaoa_max_iter": qaoa_max_iter,
        "qaoa_delta_t": qaoa_delta_t,
        "qaoa_optimizer_tol": qaoa_optimizer_tol,
        "qaoa_shots": qaoa_shots,
        "qaoa_sample_shots": qaoa_sample_shots,
        "seed": seed,
        "max_iterations": max_iterations,
        "timeout_seconds": timeout_seconds,
        "sa_t_initial": sa_t_initial,
        "sa_t_final": sa_t_final,
        "sa_alpha": sa_alpha,
        "noise_config": noise_config,
        "brute_force_max_assignments_tqudo": brute_force_max_assignments_tqudo,
        "brute_force_max_assignments_qubo": brute_force_max_assignments_qubo,
    }


def load_solver_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate solver config from YAML.

    Args:
        path: Path to YAML config file. If None, uses DEFAULT_SOLVER_CONFIG_PATH.

    Returns:
        Dict with keys: n_instances, solver, formulation, optimizer, restriction,
        qaoa_depth, qaoa_max_iter, qaoa_delta_t, qaoa_optimizer_tol, qaoa_shots,
        qaoa_sample_shots, seed, max_iterations, timeout_seconds, sa_t_initial,
        sa_t_final, sa_alpha.
        restriction is a RestrictionConfig.
        qaoa_shots controls objective-evaluation shots for all sampling-based
        QAOA backends (both QUBO and TQUDO formulations), while
        qaoa_sample_shots controls final solution sampling.

    Raises:
        ValueError: If required fields are missing or invalid.

    """
    config_path = Path(path) if path is not None else DEFAULT_SOLVER_CONFIG_PATH
    data = read_solver_yaml_as_mapping(config_path)
    return parse_solver_config_dict(data)
