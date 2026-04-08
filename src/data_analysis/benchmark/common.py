"""Shared constants and numeric helpers for benchmark comparison plots."""

from __future__ import annotations

from typing import Any

import numpy as np

_RTOL_REAL = 1e-6
_ATOL_REAL = 1e-8
_LOG_Y_FLOOR = 1e-20
# Display floor for probabilities on a log *y* axis (exact zeros are not plottable on log).
_P_OPT_LOG_AXIS_FLOOR = 1e-4


def _uniform_superposition_p_opt_htsp(n_cities: int) -> float:
    """Reference :math:`P(\\mathrm{opt})` under uniform superposition: :math:`1/(n-1)^{n-1}`."""
    if n_cities < 2:
        return float("nan")
    n1 = float(n_cities - 1)
    return 1.0 / (n1**n1)


def _uniform_superposition_p_opt_qubo(n_cities: int) -> float:
    """Reference :math:`P(\\mathrm{opt})` for QUBO under uniform :math:`|+\\rangle^{\\otimes}`: :math:`1/2^{(n-1)^2}`."""
    if n_cities < 2:
        return float("nan")
    exp = int(n_cities - 1) ** 2
    return 1.0 / (2.0**exp)


def _clip_values_for_log_y(
    vals: list[float], *, floor: float = _P_OPT_LOG_AXIS_FLOOR
) -> list[float]:
    """Same length as *vals*; non-finite and non-positive values become *floor* for log-scale drawing."""
    out: list[float] = []
    for v in vals:
        if not np.isfinite(v):
            out.append(float(floor))
            continue
        fv = float(v)
        out.append(fv if fv > float(floor) else float(floor))
    return out


def _mask_qaoa_depth_eq(qd: Any, depth: int) -> Any:
    """Boolean mask aligned with ``qd``; NaN/inf/non-numeric compare False (no int cast on inf).

    Pandas evaluates all operands in ``a & b & c`` before combining, so ``astype(int)``
    on a full column can raise even when another conjunct filters NaNs.
    """
    import pandas as pd

    q = pd.to_numeric(qd, errors="coerce")
    return q == float(depth)
