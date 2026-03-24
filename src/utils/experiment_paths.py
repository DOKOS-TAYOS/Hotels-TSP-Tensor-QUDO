"""Filesystem layout for on-disk experiment instances and solutions.

For generic output roots (raw/processed/images), see :mod:`utils.output_paths`.
"""

from __future__ import annotations

from pathlib import Path


def instances_raw_dir(output_root: Path, n_cities: int) -> Path:
    """Directory for JSON instances: ``{output}/raw/instances/n_{n_cities}``."""
    return output_root / "raw" / "instances" / f"n_{n_cities}"


def solutions_solver_root(output_root: Path, solver: str) -> Path:
    """Root for all solution trees of one backend: ``{output}/raw/solutions/{solver}``."""
    return output_root / "raw" / "solutions" / solver


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
    base = solutions_solver_root(output_root, solver) / formulation / f"n_{n_cities}"
    if qaoa_depth is not None:
        return base / str(qaoa_depth)
    return base


def instance_json_path(output_root: Path, n_cities: int, index_one_based: int) -> Path:
    """Path ``.../raw/instances/n_{n}/instance_{k}.json`` for 1-based *index_one_based*."""
    return instances_raw_dir(output_root, n_cities) / f"instance_{index_one_based}.json"
