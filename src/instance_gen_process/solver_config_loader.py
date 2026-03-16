"""Load solver configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from instance_gen_process.models import RestrictionConfig
from solvers.base import SolverRunConfig


DEFAULT_SOLVER_CONFIG_PATH = Path(__file__).with_name("solver_config.yaml")

VALID_SOLVERS = frozenset({"cudaq", "cirq", "simulated_annealing"})
VALID_FORMULATIONS = frozenset({"qubo", "tqudo"})
VALID_OPTIMIZERS = frozenset({"COBYLA", "Powell", "L-BFGS-B", "SLSQP", "Nelder-Mead"})


def load_solver_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate solver config from YAML.

    Args:
        path: Path to YAML config file. If None, uses DEFAULT_SOLVER_CONFIG_PATH.

    Returns:
        Dict with keys: n_instances, solver, formulation, optimizer, restriction,
        qaoa_depth, qaoa_max_iter, qaoa_shots, qaoa_sample_shots, seed,
        max_iterations, timeout_seconds. restriction is a RestrictionConfig.

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

    qaoa_depth = int(data.get("qaoa_depth", 1))
    qaoa_max_iter = int(data.get("qaoa_max_iter", 100))
    qaoa_shots = int(data.get("qaoa_shots", 500))
    qaoa_sample_shots = int(data.get("qaoa_sample_shots", 1000))
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)
    max_iterations = int(data.get("max_iterations", 1000))
    timeout_seconds = data.get("timeout_seconds")
    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)

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
    }


def solver_config_to_run_config(config: dict[str, Any]) -> SolverRunConfig:
    """Build SolverRunConfig from a loaded solver config dict."""
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
    )
