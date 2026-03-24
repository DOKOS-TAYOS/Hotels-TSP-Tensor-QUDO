"""Shared utility helpers (lazy exports to avoid import cycles)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "OutputLayout",
    "build_output_layout",
    "calculate_qubo_cost",
    "calculate_real_cost",
    "calculate_tqudo_cost",
    "configure_logging",
    "minimize_options",
    "serialize_instance_config",
    "serialize_restriction_config",
    "serialize_solver_result",
    "solver_config_payload_dict",
    "to_json_friendly",
    "validate_instance_constraints",
    "validate_solution_constraints_qubo",
    "validate_solution_constraints_tqudo",
]

_LAZY: dict[str, tuple[str, str]] = {
    "validate_instance_constraints": ("utils.constraints", "validate_instance_constraints"),
    "validate_solution_constraints_qubo": ("utils.constraints", "validate_solution_constraints_qubo"),
    "validate_solution_constraints_tqudo": ("utils.constraints", "validate_solution_constraints_tqudo"),
    "calculate_qubo_cost": ("utils.costs", "calculate_qubo_cost"),
    "calculate_real_cost": ("utils.costs", "calculate_real_cost"),
    "calculate_tqudo_cost": ("utils.costs", "calculate_tqudo_cost"),
    "serialize_instance_config": ("utils.experiment_serialize", "serialize_instance_config"),
    "serialize_restriction_config": ("utils.experiment_serialize", "serialize_restriction_config"),
    "serialize_solver_result": ("utils.experiment_serialize", "serialize_solver_result"),
    "solver_config_payload_dict": ("utils.experiment_serialize", "solver_config_payload_dict"),
    "to_json_friendly": ("utils.json_serialize", "to_json_friendly"),
    "configure_logging": ("utils.logging_utils", "configure_logging"),
    "minimize_options": ("utils.optimizer", "minimize_options"),
    "OutputLayout": ("utils.output_paths", "OutputLayout"),
    "build_output_layout": ("utils.output_paths", "build_output_layout"),
}


def __getattr__(name: str) -> Any:
    loc = _LAZY.get(name)
    if loc is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = loc
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


def __dir__() -> list[str]:
    return sorted(set(__all__))
