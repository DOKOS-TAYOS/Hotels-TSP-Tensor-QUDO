"""Shared scipy optimizer helpers used by all QAOA backends."""

from __future__ import annotations


def minimize_options(method: str, max_iter: int) -> dict:
    """Build scipy minimize options dict for the given method.

    Handles optimizer-specific option names so that every backend
    (Cirq, CUDA-Q, Simulated Annealing) gets a consistent configuration.

    Args:
        method: scipy optimizer name (COBYLA, Powell, L-BFGS-B, SLSQP, Nelder-Mead).
        max_iter: Maximum number of iterations / function evaluations.

    Returns:
        Dictionary suitable for ``scipy.optimize.minimize(options=...)``.
    """
    opts: dict = {"maxiter": max_iter, "disp": False}
    if method in ("Nelder-Mead", "Powell"):
        opts["maxfev"] = max_iter
    if method == "L-BFGS-B":
        opts["maxfun"] = max_iter
    return opts
