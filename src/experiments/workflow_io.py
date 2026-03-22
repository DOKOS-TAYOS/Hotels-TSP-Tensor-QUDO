"""YAML merge, instance JSON I/O, and artifact path helpers for experiment workflows."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from instance_gen_process.models import InstanceConfig, ProblemInstance

DEFAULT_INSTANCE_GENERATION_CONFIG_PATH = Path(__file__).with_name("instance_generation_config.yaml")


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


def load_instance_generation_entries(path: Path | str | None = None) -> list[tuple[int, int]]:
    """Load ``(n_cities, n_instances)`` pairs from instance generation config.

    Blocks are sorted by ``n_cities`` for stable ordering.
    """
    config_path = Path(path) if path is not None else DEFAULT_INSTANCE_GENERATION_CONFIG_PATH
    data = load_yaml_mapping(config_path)
    pairs: list[tuple[int, int]] = []
    for _block_key, block in data.items():
        if not isinstance(block, dict):
            continue
        if "n_cities" not in block or "n_instances" not in block:
            raise ValueError(
                f"Each block in {config_path} must have n_cities and n_instances; got keys: {list(block)}"
            )
        n_c = int(block["n_cities"])
        n_i = int(block["n_instances"])
        if n_i < 1:
            raise ValueError(f"n_instances must be >= 1 in block {_block_key!r}")
        pairs.append((n_c, n_i))
    pairs.sort(key=lambda t: t[0])
    return pairs


def instances_raw_dir(output_root: Path, n_cities: int) -> Path:
    """Directory for JSON instances: ``{output}/raw/instances/n_{n_cities}``."""
    return output_root / "raw" / "instances" / f"n_{n_cities}"


def solutions_raw_dir(
    output_root: Path,
    solver: str,
    formulation: str,
    n_cities: int,
    qaoa_depth: int | None,
) -> Path:
    """Directory for solution JSON under ``raw/solutions/``.

    If *qaoa_depth* is not None (Cirq/CUDA-Q), append ``/<depth>``.
    """
    base = output_root / "raw" / "solutions" / solver / formulation / f"n_{n_cities}"
    if qaoa_depth is not None:
        return base / str(qaoa_depth)
    return base


def instance_json_path(output_root: Path, n_cities: int, index_one_based: int) -> Path:
    """Path ``.../raw/instances/n_{n}/instance_{k}.json`` for 1-based *index_one_based*."""
    return instances_raw_dir(output_root, n_cities) / f"instance_{index_one_based}.json"


def serialize_problem_instance(instance: ProblemInstance) -> dict[str, Any]:
    """JSON-friendly dict for a :class:`~instance_gen_process.models.ProblemInstance`."""
    return {
        "n_cities": instance.n_cities,
        "precedences": [list(p) for p in instance.precedences],
        "prices_hotels": instance.prices_hotels.tolist(),
        "prices_travels": instance.prices_travels.tolist(),
        "seed": instance.seed,
    }


def deserialize_problem_instance(data: dict[str, Any]) -> ProblemInstance:
    """Reconstruct :class:`~instance_gen_process.models.ProblemInstance` from JSON dict."""
    raw_prec = data["precedences"]
    precedences: tuple[tuple[int, int], ...] = tuple(
        (int(p[0]), int(p[1])) for p in raw_prec
    )
    prices_hotels = np.asarray(data["prices_hotels"], dtype=np.float64)
    prices_travels = np.asarray(data["prices_travels"], dtype=np.float64)
    seed = int(data.get("seed", 0))
    return ProblemInstance(
        n_cities=int(data["n_cities"]),
        precedences=precedences,
        prices_hotels=prices_hotels,
        prices_travels=prices_travels,
        seed=seed,
    )


def load_problem_instance_json(path: Path | str) -> ProblemInstance:
    """Load a problem instance from a JSON file."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {p}")
    return deserialize_problem_instance(data)


def instance_config_for_n_cities(base: InstanceConfig, n_cities: int) -> InstanceConfig:
    """Copy *base* :class:`InstanceConfig` with a different ``n_cities``."""
    return InstanceConfig(
        n_cities=n_cities,
        n_precedences_range=base.n_precedences_range,
        prices_range_hotels=base.prices_range_hotels,
        prices_range_travels=base.prices_range_travels,
        seed=base.seed,
    )


def normalise_n_cities(value: Any) -> list[int]:
    """Coerce experiment YAML ``n_cities`` (int or list) to a list of ints."""
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(value)]


def experiment_depth_iterations(solver: str, raw: Any) -> list[tuple[int | None, int]]:
    """Pairs ``(depth_for_output_subdir, qaoa_depth_for_solver_config)``.

    For ``simulated_annealing``, the output subdir is ``None`` (no depth level);
    *raw* may be int or list (first element used as scalar for run config if list).
    For Cirq/CUDA-Q, *raw* is int or non-empty list of positive ints; path and
    run config use the same depth each time.
    """
    if solver == "simulated_annealing":
        if isinstance(raw, list):
            if not raw:
                raise ValueError("qaoa_depth list for simulated_annealing must be non-empty")
            dr = int(raw[0])
        elif raw is None:
            dr = 1
        else:
            dr = int(raw)
        if dr < 1:
            raise ValueError(f"qaoa_depth must be >= 1, got {dr}")
        return [(None, dr)]

    if raw is None:
        return [(1, 1)]
    if isinstance(raw, list):
        depths = [int(x) for x in raw]
        if not depths:
            raise ValueError("qaoa_depth list must be non-empty for quantum solvers")
        for d in depths:
            if d < 1:
                raise ValueError(f"qaoa_depth values must be >= 1, got {d}")
        return [(d, d) for d in depths]
    d = int(raw)
    if d < 1:
        raise ValueError(f"qaoa_depth must be >= 1, got {d}")
    return [(d, d)]
