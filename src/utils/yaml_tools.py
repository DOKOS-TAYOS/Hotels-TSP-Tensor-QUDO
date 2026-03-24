"""YAML loading and project-specific dict merging."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return a mapping (empty dict if file is empty)."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {p}, got {type(data).__name__}")
    return data


def read_solver_yaml_as_mapping(path: Path | str) -> dict[str, Any]:
    """Load a solver YAML file as a mapping.

    Same semantics as :func:`load_yaml_mapping` — used for
    ``load_solver_config`` and experiment YAML merge pipelines.
    """
    return load_yaml_mapping(path)


def merge_solver_yaml_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge solver-related YAML dicts; *override* wins.

    Merges nested ``restriction`` and ``noise`` mappings; other keys are
    replaced by *override* when present.
    """
    result: dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if key == "restriction" and isinstance(value, dict):
            base_r = result.get("restriction") or {}
            if not isinstance(base_r, dict):
                base_r = {}
            result["restriction"] = {**base_r, **value}
        elif key == "noise" and isinstance(value, dict):
            base_n = result.get("noise") or {}
            if not isinstance(base_n, dict):
                base_n = {}
            result["noise"] = {**base_n, **value}
        else:
            result[key] = deepcopy(value)
    return result
