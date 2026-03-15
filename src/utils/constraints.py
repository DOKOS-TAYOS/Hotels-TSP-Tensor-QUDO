"""Constraint validation helpers for aircraft loading instances."""

from __future__ import annotations

from instance_gen_process import ProblemInstance


def validate_instance_constraints(instance: ProblemInstance) -> bool:
    """Return `True` when basic instance constraints are consistent."""

    if instance.max_weight <= 0 or instance.max_volume <= 0:
        return False
    if instance.cg_min >= instance.cg_max:
        return False

    for item in instance.items:
        if item.weight <= 0 or item.volume <= 0:
            return False

    return True

def idx(t: int, i: int, n_available: int):
    return t*n_available + i