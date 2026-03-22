"""SIGINT flag for Cirq QAOA (slow per-shot simulation).

CUDA-Q is left unchanged. The workflow calls :func:`request_solver_stop` on
Ctrl-C; Cirq ``cost_fn`` checks :func:`raise_if_solver_stop_requested` between
evaluations (not inside a running ``simulator.run`` call).
"""

from __future__ import annotations

_stop_requested: bool = False


class SolverStopRequested(Exception):
    """User requested stop (e.g. SIGINT) during a solver run."""


def request_solver_stop() -> None:
    """Set the cooperative stop flag (typically from a SIGINT handler)."""
    global _stop_requested
    _stop_requested = True


def clear_solver_stop_request() -> None:
    """Clear the flag at the start of a workflow or batch run."""
    global _stop_requested
    _stop_requested = False


def is_solver_stop_requested() -> bool:
    """Return True if :func:`request_solver_stop` was called."""
    return _stop_requested


def raise_if_solver_stop_requested() -> None:
    """Raise :class:`SolverStopRequested` if a stop was requested."""
    if _stop_requested:
        raise SolverStopRequested
