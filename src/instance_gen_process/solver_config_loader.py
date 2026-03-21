"""Load solver configuration from YAML files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from instance_gen_process.models import InstanceConfig, RestrictionConfig
from utils.qaoa_helpers import is_power_of_two

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from solvers.base import SolverRunConfig


DEFAULT_SOLVER_CONFIG_PATH = Path(__file__).with_name("solver_config.yaml")

VALID_SOLVERS = frozenset({"cudaq", "cirq", "simulated_annealing"})
VALID_FORMULATIONS = frozenset({"qubo", "tqudo", "tqudo_virtual"})
VALID_OPTIMIZERS = frozenset({"COBYLA", "Powell", "L-BFGS-B", "SLSQP", "Nelder-Mead"})


def _parse_int_setting(raw_value: Any, field_name: str, minimum: int) -> int:
    """Parse an integer setting and enforce a minimum value."""
    value = int(raw_value)
    if value < minimum:
        comparator = "at least" if minimum > 0 else "greater than or equal to"
        raise ValueError(f"{field_name} must be {comparator} {minimum}")
    return value


def _validate_cobyla_budget(qaoa_depth: int, qaoa_max_iter: int, optimizer: str) -> None:
    """Ensure COBYLA receives enough evaluations for the QAOA parameter count."""
    if optimizer != "COBYLA":
        return

    minimum_iterations = (2 * qaoa_depth) + 2
    if qaoa_max_iter < minimum_iterations:
        raise ValueError(
            "qaoa_max_iter is too small for COBYLA. "
            f"With qaoa_depth={qaoa_depth}, it must be at least {minimum_iterations}."
        )


_QUBO_QUBIT_WARN_THRESHOLD = 30


def validate_solver_instance_compatibility(
    instance_config: InstanceConfig,
    solver_config: dict[str, Any],
) -> None:
    """Validate constraints that depend on both instance and solver configuration.

    Backend capability matrix:

    - Cirq:  qubo, tqudo (native qudits), tqudo_virtual (qubit emulation)
    - CUDA-Q: qubo, tqudo_virtual
    - SA:    qubo, tqudo
    """
    formulation = solver_config.get("formulation", "tqudo")
    solver = solver_config.get("solver", "cudaq")
    n_available = instance_config.n_cities - 1

    if formulation == "qubo" and solver in ("cudaq", "cirq"):
        n_qubits = n_available * n_available
        if n_qubits > _QUBO_QUBIT_WARN_THRESHOLD:
            raise ValueError(
                f"QUBO formulation with {instance_config.n_cities} cities requires "
                f"{n_qubits} qubits for quantum simulation, which exceeds the "
                f"practical limit (~{_QUBO_QUBIT_WARN_THRESHOLD}). "
                "Use formulation='tqudo' or 'tqudo_virtual' for quantum backends, "
                "or solver='simulated_annealing' for QUBO at this scale."
            )

    if formulation == "tqudo" and solver == "cudaq":
        raise ValueError(
            "Native TQUDO (real qudits) is not supported by the CUDA-Q backend. "
            "Use formulation='tqudo_virtual' for qubit-emulated TQUDO on CUDA-Q, "
            "or use solver='cirq' for native qudit support."
        )

    if formulation == "tqudo_virtual" and solver == "simulated_annealing":
        raise ValueError(
            "TQUDO virtual (qubit emulation) is not supported by simulated annealing. "
            "Use formulation='tqudo' or 'qubo' instead."
        )

    if formulation == "tqudo_virtual" and not is_power_of_two(n_available):
        raise ValueError(
            f"TQUDO virtual (qubit emulation) requires n_cities - 1 to be a "
            "power of two. Use formulation='tqudo' with solver='cirq' (native "
            "qudits) for arbitrary dimensions. "
            f"Got n_cities={instance_config.n_cities} (n_cities - 1 = {n_available})."
        )


def load_solver_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate solver config from YAML.

    Args:
        path: Path to YAML config file. If None, uses DEFAULT_SOLVER_CONFIG_PATH.

    Returns:
        Dict with keys: n_instances, solver, formulation, optimizer, restriction,
        qaoa_depth, qaoa_max_iter, qaoa_shots, qaoa_sample_shots, seed,
        max_iterations, timeout_seconds, sa_t_initial, sa_t_final, sa_alpha.
        restriction is a RestrictionConfig.
        qaoa_shots controls objective-evaluation shots for all sampling-based
        QAOA backends (both QUBO and TQUDO formulations), while
        qaoa_sample_shots controls final solution sampling.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    config_path = Path(path) if path is not None else DEFAULT_SOLVER_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

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
        "qaoa_shots": qaoa_shots,
        "qaoa_sample_shots": qaoa_sample_shots,
        "seed": seed,
        "max_iterations": max_iterations,
        "timeout_seconds": timeout_seconds,
        "sa_t_initial": sa_t_initial,
        "sa_t_final": sa_t_final,
        "sa_alpha": sa_alpha,
        "noise_config": noise_config,
    }


def solver_config_to_run_config(config: dict[str, Any]) -> SolverRunConfig:
    """Build SolverRunConfig from a loaded solver config dict."""
    from solvers.base import SolverRunConfig
    from solvers.noise import NoiseConfig

    return SolverRunConfig(
        max_iterations=config["max_iterations"],
        timeout_seconds=config["timeout_seconds"],
        formulation=config["formulation"],
        restriction_config=config["restriction"],
        qaoa_depth=config["qaoa_depth"],
        qaoa_max_iter=config["qaoa_max_iter"],
        qaoa_shots=config["qaoa_shots"],
        qaoa_sample_shots=config["qaoa_sample_shots"],
        seed=config["seed"],
        optimizer=config["optimizer"],
        noise_config=config.get("noise_config", NoiseConfig()),
        sa_t_initial=config["sa_t_initial"],
        sa_t_final=config["sa_t_final"],
        sa_alpha=config["sa_alpha"],
    )
