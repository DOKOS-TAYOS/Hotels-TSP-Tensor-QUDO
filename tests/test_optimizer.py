"""Contract tests for ``utils.optimizer.minimize_options`` (SciPy minimize ``options=``)."""

from __future__ import annotations

import pytest

from utils.optimizer import minimize_options


@pytest.mark.parametrize(
    ("method", "expected_extra_keys"),
    [
        ("Nelder-Mead", {"maxfev", "xatol", "fatol"}),
        ("nelder-mead", {"maxfev", "xatol", "fatol"}),
        ("Powell", {"maxfev", "xtol", "ftol"}),
        ("powell", {"maxfev", "xtol", "ftol"}),
        ("L-BFGS-B", {"maxfun", "ftol", "gtol"}),
        ("l-bfgs-b", {"maxfun", "ftol", "gtol"}),
        ("SLSQP", {"ftol"}),
        ("slsqp", {"ftol"}),
        ("COBYLA", {"tol"}),
        ("cobyla", {"tol"}),
    ],
)
def test_minimize_options_known_methods(
    method: str,
    expected_extra_keys: set[str],
) -> None:
    max_iter = 100
    tol = 1e-5
    opts = minimize_options(method, max_iter=max_iter, tol=tol)

    assert opts["maxiter"] == max_iter
    assert opts["disp"] is False
    for key in expected_extra_keys:
        assert key in opts
        assert isinstance(opts[key], (float, int))

    if "maxfev" in expected_extra_keys:
        assert opts["maxfev"] == max_iter
    if "maxfun" in expected_extra_keys:
        assert opts["maxfun"] == max_iter


def test_minimize_options_unknown_method_gets_baseline_only() -> None:
    opts = minimize_options("unknown-heuristic", max_iter=50, tol=0.01)
    assert opts == {"maxiter": 50, "disp": False}
