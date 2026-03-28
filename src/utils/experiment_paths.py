"""Filesystem layout for on-disk experiment instances and solutions.

For generic output roots (raw/processed/images), see :mod:`utils.output_paths`.
"""

from __future__ import annotations

from pathlib import Path


def instances_raw_dir(output_root: Path, n_cities: int) -> Path:
    """Return the directory for generated instance JSON files.

    Args:
        output_root: Experiment output root (e.g. ``output/``).
        n_cities: City count label used in the folder name.

    Returns:
        Path ``{output_root}/raw/instances/n_{n_cities}``.
    """
    return output_root / "raw" / "instances" / f"n_{n_cities}"


def solutions_solver_root(output_root: Path, solver: str) -> Path:
    """Return the root directory for one solver's solution tree.

    Args:
        output_root: Experiment output root.
        solver: Backend name (e.g. ``cudaq``, ``cirq``).

    Returns:
        Path ``{output_root}/raw/solutions/{solver}``.
    """
    return output_root / "raw" / "solutions" / solver


def solutions_raw_dir(
    output_root: Path,
    solver: str,
    formulation: str,
    n_cities: int,
    qaoa_depth: int | None,
) -> Path:
    """Return the directory for solution JSON for a solver/formulation/n_cities run.

    Args:
        output_root: Experiment output root.
        solver: Backend name.
        formulation: ``qubo``, ``tqudo``, or ``tqudo_virtual``.
        n_cities: City count in the path segment ``n_{n_cities}``.
        qaoa_depth: If not None (QAOA backends), appended as a final path segment.

    Returns:
        Path under ``raw/solutions/`` including formulation and optional depth.
    """
    base = solutions_solver_root(output_root, solver) / formulation / f"n_{n_cities}"
    if qaoa_depth is not None:
        return base / str(qaoa_depth)
    return base


def instance_json_path(output_root: Path, n_cities: int, index_one_based: int) -> Path:
    """Return the path to a single on-disk instance JSON file.

    Args:
        output_root: Experiment output root.
        n_cities: City count for the instances folder.
        index_one_based: 1-based instance index ``k`` in ``instance_{k}.json``.

    Returns:
        Path to ``instance_{k}.json`` under the instances directory.
    """
    return instances_raw_dir(output_root, n_cities) / f"instance_{index_one_based}.json"
