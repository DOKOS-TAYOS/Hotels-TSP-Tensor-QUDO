"""JSON-friendly snapshots for experiment outputs and metadata."""

from __future__ import annotations

from typing import Any

from instance_gen_process.models import InstanceConfig, RestrictionConfig
from solvers.base import SolverResult

from utils.json_serialize import to_json_friendly


def serialize_instance_config(config: InstanceConfig) -> dict[str, Any]:
    """Convert :class:`~instance_gen_process.models.InstanceConfig` to JSON-friendly dict."""
    return {
        "n_cities": config.n_cities,
        "n_precedences_range": list(config.n_precedences_range),
        "prices_range_hotels": list(config.prices_range_hotels),
        "prices_range_travels": list(config.prices_range_travels),
        "seed": config.seed,
    }


def serialize_restriction_config(restriction: RestrictionConfig) -> dict[str, float]:
    """Convert :class:`~instance_gen_process.models.RestrictionConfig` to plain floats."""
    return {
        "lambda_0": restriction.lambda_0,
        "lambda_1": restriction.lambda_1,
        "lambda_2": restriction.lambda_2,
    }


def serialize_solver_result(result: SolverResult) -> dict[str, Any]:
    """Convert :class:`~solvers.base.SolverResult` to a JSON-friendly dict."""
    return {
        "solver_name": result.solver_name,
        "objective_value": result.objective_value,
        "feasible": result.feasible,
        "runtime_seconds": result.runtime_seconds,
        "metadata": to_json_friendly(result.metadata),
    }


def build_solution_record(
    *,
    instance: dict[str, Any],
    instance_config: dict[str, Any],
    instance_index: int,
    solver_config: dict[str, Any],
    solver_output: dict[str, Any],
    instance_source: str | None = None,
) -> dict[str, Any]:
    """Assemble the standard top-level JSON object for experiment solution files."""
    out: dict[str, Any] = {
        "instance": instance,
        "instance_config": instance_config,
        "instance_index": instance_index,
        "solver_config": solver_config,
        "solver_output": solver_output,
    }
    if instance_source is not None:
        out["instance_source"] = instance_source
    return out


def solver_config_payload_dict(solver_config_dict: dict[str, Any]) -> dict[str, Any]:
    """Build JSON-safe solver config snapshot (expand restriction dataclass)."""
    restriction = solver_config_dict["restriction"]
    serializable: dict[str, Any] = {
        k: v for k, v in solver_config_dict.items() if k != "restriction"
    }
    serializable["restriction"] = {
        "lambda_0": restriction.lambda_0,
        "lambda_1": restriction.lambda_1,
        "lambda_2": restriction.lambda_2,
    }
    return to_json_friendly(serializable)
