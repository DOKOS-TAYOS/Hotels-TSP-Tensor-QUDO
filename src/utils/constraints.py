"""Constraint validation helpers for travel routing (Hotel TSP) instances."""

from __future__ import annotations

from collections import deque

import numpy as np

from instance_gen_process.models import ProblemInstance


def idx(t: int, i: int, n_available: int) -> int:
    """Return the flat QUBO index for timestep *t* and city *i*.

    Args:
        t: Timestep in ``{0, …, n_available - 1}``.
        i: City index in ``{0, …, n_available - 1}``.
        n_available: Number of non-depot cities.

    Returns:
        Linear index ``t * n_available + i`` into the one-hot vectorisation.

    """
    return t * n_available + i


def validate_instance_constraints(instance: ProblemInstance) -> bool:
    """Check shapes, index ranges, and acyclicity of precedence constraints.

    Args:
        instance: Problem instance to validate.

    Returns:
        True if dimensions and precedences are self-consistent.

    """
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


def _has_cycle(
    precedences: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    n_nodes: int,
) -> bool:
    """Return whether the directed precedence graph has a cycle.

    Args:
        precedences: Directed edges (origin, destination) meaning origin before destination.
        n_nodes: Number of nodes (cities) in ``{0, …, n_nodes - 1}``.

    Returns:
        True if a directed cycle exists.

    """
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
    precedences: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: int,
    destination: int,
) -> bool:
    """Return whether edge (origin, destination) would close a directed cycle.

    A cycle occurs when *destination* can already reach *origin* in the graph
    formed by *precedences*. Uses BFS from *destination*.

    Args:
        precedences: Current precedence edges.
        origin: Proposed edge tail (must visit earlier).
        destination: Proposed edge head (must visit later).

    Returns:
        True if adding the edge would introduce a cycle.

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


def _check_precedences(
    seq: np.ndarray,
    precedences: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bool:
    """Return whether every precedence pair appears in order in *seq*.

    Args:
        seq: Route as city indices per timestep.
        precedences: Pairs ``(a, b)`` requiring *a* before *b*.

    Returns:
        True if all precedences are satisfied.

    """
    pos: dict[int, int] = {int(seq[t]): t for t in range(len(seq))}
    return all(a in pos and b in pos and pos[a] < pos[b] for a, b in precedences)


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

    if len(set(seq)) != n_available:
        return False

    if not all(0 <= x < n_available for x in seq):
        return False

    return _check_precedences(seq, instance.precedences)


def qubo_binary_to_sequence(solution: np.ndarray, n_available: int) -> np.ndarray | None:
    """Decode a QUBO binary vector to a city-per-timestep sequence.

    Args:
        solution: Flat binary vector of length ``n_available ** 2``.
        n_available: Number of cities (excluding depot).

    Returns:
        Integer array of shape ``(n_available,)`` with the route, or None if
        the one-hot constraints are violated.

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

    return _check_precedences(seq, instance.precedences)
