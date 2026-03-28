"""JSON-friendly snapshots for experiment outputs and metadata."""

from __future__ import annotations

from typing import Any

from instance_gen_process.models import InstanceConfig, RestrictionConfig
from solvers.base import SolverResult

from utils.json_serialize import to_json_friendly


def serialize_instance_config(config: InstanceConfig) -> dict[str, Any]:
    """Convert ``InstanceConfig`` to a JSON-serialisable dict.

    Args:
        config: Instance generation parameters.

    Returns:
        Mapping with lists for tuple fields, suitable for ``json.dump``.
    """
    return {
        "n_cities": config.n_cities,
        "n_precedences_range": list(config.n_precedences_range),
        "prices_range_hotels": list(config.prices_range_hotels),
        "prices_range_travels": list(config.prices_range_travels),
        "seed": config.seed,
    }


def serialize_restriction_config(restriction: RestrictionConfig) -> dict[str, float]:
    """Convert ``RestrictionConfig`` to a flat dict of penalty floats.

    Args:
        restriction: QUBO/TQUDO penalty coefficients.

    Returns:
        Dict with keys ``lambda_0``, ``lambda_1``, ``lambda_2``.
    """
    return {
        "lambda_0": restriction.lambda_0,
        "lambda_1": restriction.lambda_1,
        "lambda_2": restriction.lambda_2,
    }


def serialize_solver_result(result: SolverResult) -> dict[str, Any]:
    """Convert ``SolverResult`` to a JSON-safe dict.

    Args:
        result: Solver output from any backend.

    Returns:
        Dict with objective, feasibility, runtime, and JSON-normalised metadata.
    """
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
    """Assemble the canonical on-disk solution JSON object.

    Args:
        instance: Serialised ``ProblemInstance`` snapshot.
        instance_config: Serialised instance-generation parameters.
        instance_index: Zero-based index within the batch for the run.
        solver_config: Merged solver settings snapshot (JSON-safe).
        solver_output: Serialised solver result or error payload.
        instance_source: Optional path to the input instance JSON.

    Returns:
        Dict written as one ``instance_*.json`` under ``raw/solutions/``.
    """
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
    """Build a JSON-safe copy of a validated solver config dict.

    Args:
        solver_config_dict: Mapping from ``parse_solver_config_dict`` /
            ``load_solver_config`` (includes ``restriction`` dataclass).

    Returns:
        Shallow copy with ``restriction`` expanded to plain floats, then passed
        through ``to_json_friendly``.
    """
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
