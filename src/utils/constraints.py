"""Constraint validation helpers for travel routing (Hotel TSP) instances."""

from __future__ import annotations

from instance_gen_process.models import ProblemInstance


def idx(t: int, i: int, n_available: int) -> int:
    """Linear index for (timestep, city) in flattened QUBO representation."""
    return t * n_available + i


def validate_instance_constraints(instance: ProblemInstance) -> bool:
    """Return `True` when basic instance constraints are consistent."""

    n_available = instance.n_cities - 1
    if n_available < 1:
        return False

    if instance.prices_hotels.shape != (n_available, n_available):
        return False
    if instance.prices_travels.shape != (n_available, n_available, n_available):
        return False

    for a, b in instance.precedences:
        if not (0 <= a < n_available and 0 <= b < n_available):
            return False
        if a == b:
            return False

    if _has_cycle(instance.precedences, n_available):
        return False

    return True


def _has_cycle(precedences: list[tuple[int, int]], n_nodes: int) -> bool:
    """True if the precedence graph contains a cycle."""
    adj: dict[int, list[int]] = {}
    for a, b in precedences:
        adj.setdefault(a, []).append(b)

    visited: set[int] = set()
    rec_stack: set[int] = set()

    def dfs(node: int) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.discard(node)
        return False

    for node in range(n_nodes):
        if node not in visited and dfs(node):
            return True
    return False
