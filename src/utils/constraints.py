"""Constraint validation helpers for travel routing (Hotel TSP) instances."""

from __future__ import annotations

from collections import deque

import numpy as np

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
    if instance.prices_travels.shape != (instance.n_cities, instance.n_cities, instance.n_cities):
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


def would_create_cycle(
    precedences: list[tuple[int, int]], origin: int, destination: int
) -> bool:
    """True if adding (origin, destination) would create a cycle in the precedence graph.

    A cycle occurs when destination can already reach origin (directly or via other rules).
    Uses BFS from destination with deque.popleft for O(1) queue operations.
    """
    adj: dict[int, list[int]] = {}
    for a, b in precedences:
        adj.setdefault(a, []).append(b)
    seen: set[int] = {destination}
    queue: deque[int] = deque([destination])
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor == origin:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return False


def validate_solution_constraints_tqudo(
    instance: ProblemInstance,
    solution: list[int] | np.ndarray,
) -> bool:
    """Check that a T-QUDO solution satisfies precedence and no-duplicate constraints.

    - Precedence: for each (a, b) in instance.precedences, a appears before b.
    - No duplicates: the solution contains each node at most once.

    Args:
        instance: The problem instance with precedences and n_cities.
        solution: Sequence of length n_available where solution[t] = city at timestep t.

    Returns:
        True if all constraints are satisfied, False otherwise.
    """
    n_available = instance.n_cities - 1
    seq = np.asarray(solution).flatten()

    if len(seq) != n_available:
        return False

    # No duplicates: all elements must be unique
    if len(set(seq)) != n_available:
        return False

    # All nodes must be in valid range
    if not all(0 <= x < n_available for x in seq):
        return False

    # Precedence: precedence[0] must appear before precedence[1]
    pos: dict[int, int] = {int(seq[t]): t for t in range(n_available)}
    for a, b in instance.precedences:
        if a not in pos or b not in pos:
            return False
        if pos[a] >= pos[b]:
            return False

    return True


def qubo_binary_to_sequence(solution: np.ndarray, n_available: int) -> np.ndarray | None:
    """Decode QUBO binary vector to a sequence of cities per timestep.

    Returns None if the binary encoding is invalid (not exactly one 1 per row/col).
    """
    x = np.asarray(solution).flatten()
    expected_len = n_available * n_available
    if len(x) != expected_len:
        return None

    seq = np.full(n_available, -1, dtype=int)
    for t in range(n_available):
        count = 0
        chosen = -1
        for i in range(n_available):
            if x[idx(t, i, n_available)] > 0.5:  # Treat as active
                count += 1
                chosen = i
        if count != 1:
            return None
        seq[t] = chosen

    # Check each city appears exactly once
    if len(set(seq)) != n_available:
        return None
    return seq


def sequence_to_qubo_binary(
    sequence: np.ndarray | list[int],
    n_available: int,
) -> np.ndarray:
    """Encode a route sequence to QUBO binary vector (one-hot per timestep).

    Args:
        sequence: Route as list/array where sequence[t] = city at timestep t.
        n_available: Number of available cities (n_cities - 1).

    Returns:
        Binary vector of shape (n_available * n_available,) with one-hot encoding.
    """
    seq = np.asarray(sequence).flatten()
    x = np.zeros(n_available * n_available, dtype=float)
    for t in range(n_available):
        city = int(seq[t])
        x[idx(t, city, n_available)] = 1.0
    return x


def validate_solution_constraints_qubo(
    instance: ProblemInstance,
    solution: np.ndarray,
) -> bool:
    """Check that a QUBO solution satisfies precedence and no-duplicate constraints.

    Decodes the binary QUBO vector to a sequence, then validates:
    - Precedence: for each (a, b) in instance.precedences, a appears before b.
    - No duplicates: exactly one city per timestep and one timestep per city.

    Args:
        instance: The problem instance with precedences and n_cities.
        solution: Binary vector of shape (n_available * n_available,).

    Returns:
        True if all constraints are satisfied, False otherwise.
    """
    n_available = instance.n_cities - 1
    seq = qubo_binary_to_sequence(solution, n_available)
    if seq is None:
        return False

    # Precedence: precedence[0] must appear before precedence[1]
    pos: dict[int, int] = {int(seq[t]): t for t in range(n_available)}
    for a, b in instance.precedences:
        if a not in pos or b not in pos:
            return False
        if pos[a] >= pos[b]:
            return False

    return True

