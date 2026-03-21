"""Shared test helpers and fixtures."""

from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from instance_gen_process.models import ProblemInstance, RestrictionConfig


# ---------------------------------------------------------------------------
# Workspace temp directory helpers
# ---------------------------------------------------------------------------


def workspace_tmp_dir(prefix: str) -> Path:
    """Create a temporary directory under tests/.tmp for isolated test runs."""
    base_dir = Path(__file__).resolve().parent / ".tmp"
    base_dir.mkdir(exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def cleanup_workspace_tmp_dir(temp_dir: Path) -> None:
    """Remove a temporary directory and its parent if empty."""
    shutil.rmtree(temp_dir, ignore_errors=True)
    base_dir = temp_dir.parent
    if base_dir.exists() and not any(base_dir.iterdir()):
        base_dir.rmdir()


# ---------------------------------------------------------------------------
# Instance factories
# ---------------------------------------------------------------------------


def make_problem_instance(
    n_cities: int = 4,
    precedences: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    prices_hotels: np.ndarray | None = None,
    prices_travels: np.ndarray | None = None,
    seed: int = 0,
) -> ProblemInstance:
    """Build a ProblemInstance with sensible defaults for testing."""
    n_available = n_cities - 1
    if prices_hotels is None:
        prices_hotels = np.ones((n_available, n_available), dtype=float)
    if prices_travels is None:
        prices_travels = np.ones((n_cities, n_cities, n_cities), dtype=float)
        for k in range(n_cities):
            prices_travels[:, k, k] = 0.0
    return ProblemInstance(
        n_cities=n_cities,
        precedences=tuple(precedences) if precedences else (),
        prices_hotels=prices_hotels,
        prices_travels=prices_travels,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tensor factories
# ---------------------------------------------------------------------------


def synthetic_tqudo_tensors(
    n_qudits: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build small sparse tensors that exercise both local and long-range phases."""
    Etab = np.zeros((n_qudits, dimension, dimension), dtype=float)
    Ettprimeab = np.zeros(
        (n_qudits, n_qudits, dimension, dimension),
        dtype=float,
    )
    Etab[0, 0, dimension - 1] = 1.0
    if n_qudits > 2:
        Etab[1, min(1, dimension - 1), max(0, dimension - 2)] = 0.5
        Ettprimeab[0, n_qudits - 1, dimension - 1, 0] = 0.75
    return Etab, Ettprimeab


# ---------------------------------------------------------------------------
# Restriction configs
# ---------------------------------------------------------------------------


def high_penalty_restriction() -> RestrictionConfig:
    """Penalties much larger than any feasible cost to test constraint enforcement."""
    return RestrictionConfig(lambda_0=1000.0, lambda_1=1000.0, lambda_2=1000.0)


def zero_penalty_restriction() -> RestrictionConfig:
    """Zero penalties to isolate cost terms from constraint penalties."""
    return RestrictionConfig(lambda_0=0.0, lambda_1=0.0, lambda_2=0.0)
