"""Shared scipy optimizer helpers used by all QAOA backends."""

from __future__ import annotations


def minimize_options(
    method: str,
    max_iter: int,
    tol: float = 1e-6,
) -> dict[str, float | int | bool]:
    """Build a SciPy ``minimize`` ``options`` dict for *method*.

    Maps a unified iteration budget onto method-specific keys (``maxiter``,
    ``maxfev``, ``maxfun``) so Cirq, CUDA-Q, and simulated annealing callers
    stay consistent. Tolerance keys mirror SciPy's handling of the top-level
    ``tol`` argument for each method (see :func:`scipy.optimize.minimize`).

    Args:
        method: SciPy method name (``COBYLA``, ``Powell``, ``L-BFGS-B``,
            ``SLSQP``, ``Nelder-Mead``).
        max_iter: Cap on iterations or function evaluations.
        tol: Stopping tolerance (function / parameter scales depend on the method).

    Returns:
        Dict passed as ``options=`` to :func:`scipy.optimize.minimize`.
    """
    opts: dict[str, float | int | bool] = {"maxiter": max_iter, "disp": False}
    if method in ("Nelder-Mead", "Powell"):
        opts["maxfev"] = max_iter
    if method == "L-BFGS-B":
        opts["maxfun"] = max_iter

    meth = method.lower()
    if meth == "nelder-mead":
        opts["xatol"] = tol
        opts["fatol"] = tol
    elif meth == "powell":
        opts["xtol"] = tol
        opts["ftol"] = tol
    elif meth == "l-bfgs-b":
        opts["ftol"] = tol
        opts["gtol"] = tol
    elif meth == "slsqp":
        opts["ftol"] = tol
    elif meth == "cobyla":
        opts["tol"] = tol

    return opts
