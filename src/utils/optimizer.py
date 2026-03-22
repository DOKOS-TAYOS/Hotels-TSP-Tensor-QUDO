"""Shared scipy optimizer helpers used by all QAOA backends."""

from __future__ import annotations


def minimize_options(method: str, max_iter: int) -> dict:
    """Build a SciPy ``minimize`` ``options`` dict for *method*.

    Maps a unified iteration budget onto method-specific keys (``maxiter``,
    ``maxfev``, ``maxfun``) so Cirq, CUDA-Q, and simulated annealing callers
    stay consistent.

    Args:
        method: SciPy method name (``COBYLA``, ``Powell``, ``L-BFGS-B``,
            ``SLSQP``, ``Nelder-Mead``).
        max_iter: Cap on iterations or function evaluations.

    Returns:
        Dict passed as ``options=`` to :func:`scipy.optimize.minimize`.
    """
    opts: dict = {"maxiter": max_iter, "disp": False}
    if method in ("Nelder-Mead", "Powell"):
        opts["maxfev"] = max_iter
    if method == "L-BFGS-B":
        opts["maxfun"] = max_iter
    return opts
