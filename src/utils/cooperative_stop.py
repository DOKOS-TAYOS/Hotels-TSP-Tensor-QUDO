"""SIGINT flag for Cirq QAOA (slow per-shot simulation).

CUDA-Q is left unchanged. The workflow calls :func:`request_solver_stop` on
Ctrl-C; Cirq ``cost_fn`` checks :func:`raise_if_solver_stop_requested` between
evaluations (not inside a running ``simulator.run`` call).
"""

from __future__ import annotations

_stop_requested: bool = False


class SolverStopRequested(Exception):
    """Raised when the user requests a cooperative stop during solving."""


def request_solver_stop() -> None:
    """Set the global cooperative stop flag (e.g. from a SIGINT handler)."""
    global _stop_requested
    _stop_requested = True


def clear_solver_stop_request() -> None:
    """Reset the stop flag before a new workflow or batch."""
    global _stop_requested
    _stop_requested = False


def is_solver_stop_requested() -> bool:
    """Return whether :func:`request_solver_stop` has been called since last clear."""
    return _stop_requested


def raise_if_solver_stop_requested() -> None:
    """Raise :exc:`SolverStopRequested` if cooperative stop was requested.

    Raises:
        SolverStopRequested: If :func:`request_solver_stop` was called.

    """
    if _stop_requested:
        raise SolverStopRequested
