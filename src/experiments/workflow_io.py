"""YAML merge, instance JSON I/O, and artifact path helpers for experiment workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from instance_gen_process.models import InstanceConfig, ProblemInstance

from utils.experiment_paths import (
    instance_json_path,
    instances_raw_dir,
    solutions_raw_dir,
    solutions_solver_root,
)
from utils.yaml_tools import load_yaml_mapping

DEFAULT_INSTANCE_GENERATION_CONFIG_PATH = Path(__file__).with_name("instance_generation_config.yaml")


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

    For ``simulated_annealing`` and ``brute_force``, the output subdir is ``None``
    (no depth level); *raw* may be int or list (first element used as scalar for
    run config if list).
    For Cirq/CUDA-Q, *raw* is int or non-empty list of positive ints; path and
    run config use the same depth each time.
    """
    if solver in ("simulated_annealing", "brute_force"):
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


__all__ = [
    "DEFAULT_INSTANCE_GENERATION_CONFIG_PATH",
    "deserialize_problem_instance",
    "experiment_depth_iterations",
    "instance_config_for_n_cities",
    "instance_json_path",
    "instances_raw_dir",
    "load_instance_generation_entries",
    "load_problem_instance_json",
    "normalise_n_cities",
    "serialize_problem_instance",
    "solutions_raw_dir",
    "solutions_solver_root",
]
