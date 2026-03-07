"""Data models for generated aircraft loading instances."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CargoItem:
    """Single cargo item with physical properties used by solvers."""

    item_id: int
    weight: float
    volume: float


@dataclass(frozen=True, slots=True)
class InstanceConfig:
    """Configuration that controls random instance generation."""

    num_items: int
    max_weight: float
    max_volume: float
    cg_min: float
    cg_max: float
    weight_range: tuple[float, float] = (100.0, 1000.0)
    volume_range: tuple[float, float] = (1.0, 10.0)
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ProblemInstance:
    """Canonical in-memory problem representation consumed by solvers."""

    items: tuple[CargoItem, ...]
    max_weight: float
    max_volume: float
    cg_min: float
    cg_max: float
