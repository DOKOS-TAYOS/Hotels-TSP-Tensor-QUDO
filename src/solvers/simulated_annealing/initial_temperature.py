"""Estimate the initial temperature T₀ for simulated annealing (Ben-Ameur method).

Samples uphill transitions on a given problem instance, then iteratively
refines a trial temperature until the estimated acceptance ratio matches a
target χ₀ within tolerance ε.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from instance_gen_process import (
    generate_QUBO_from_problem,
    generate_TQUDO_from_problem,
)
from instance_gen_process.models import (
    ProblemInstance,
    ProblemQUBO,
    ProblemTQUDO,
    RestrictionConfig,
)
from solvers.simulated_annealing.solver import (
    _default_restriction,
    evaluate_cost,
    random_neighbor,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T0EstimationResult:
    """Result of the initial temperature estimation procedure.

    Attributes:
        t0: Estimated initial temperature.
        chi_achieved: Acceptance ratio achieved at *t0*.
        iterations: Number of iterative refinement steps taken.
        n_samples: Number of uphill transitions used.
        converged: Whether the algorithm converged within tolerance.

    """

    t0: float
    chi_achieved: float
    iterations: int
    n_samples: int
    converged: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_FALLBACK_T0 = 1.0


def _collect_uphill_transitions(
    formulation: str,
    problem: ProblemQUBO | ProblemTQUDO,
    n_available: int,
    n_samples: int,
    rng: np.random.Generator,
    max_attempts_factor: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample strictly-positive cost transitions (δ > 0).

    Returns parallel arrays ``(e_min, e_max)`` of length ``<= n_samples``
    where ``e_max[i] - e_min[i] > 0`` for every *i*.
    """
    e_min_list: list[float] = []
    e_max_list: list[float] = []

    current = rng.permutation(n_available).astype(np.int64)
    current_cost = evaluate_cost(formulation, problem, current, n_available)

    max_attempts = max_attempts_factor * n_samples
    attempts = 0

    while len(e_min_list) < n_samples and attempts < max_attempts:
        neighbor, _ = random_neighbor(current, rng)
        neighbor_cost = evaluate_cost(formulation, problem, neighbor, n_available)
        delta = neighbor_cost - current_cost

        if delta > 0:
            e_min_list.append(current_cost)
            e_max_list.append(neighbor_cost)

        # Accept the neighbor with 50 % probability to diversify sampling.
        if rng.random() < 0.5:
            current = neighbor
            current_cost = neighbor_cost

        attempts += 1

    return np.asarray(e_min_list, dtype=np.float64), np.asarray(e_max_list, dtype=np.float64)


def _compute_acceptance_ratio(
    e_min: np.ndarray,
    e_max: np.ndarray,
    temperature: float,
) -> float:
    r"""Estimate acceptance ratio at *temperature*.

    .. math::

        \hat{\chi}(T) = \frac{\sum \exp(-E^{\max}_t / T)}
                              {\sum \exp(-E^{\min}_t / T)}

    Uses a shift for numerical stability.
    """
    shift = float(np.min(e_min))
    numerator = float(np.sum(np.exp(-(e_max - shift) / temperature)))
    denominator = float(np.sum(np.exp(-(e_min - shift) / temperature)))
    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def _refine_temperature(
    e_min: np.ndarray,
    e_max: np.ndarray,
    chi_0: float,
    t_initial: float,
    epsilon: float,
    p_initial: float,
    max_iter: int,
) -> tuple[float, float, int, bool]:
    """Refine *t_initial* iteratively until χ̂(T) ≈ χ₀.

    Returns ``(temperature, chi_achieved, iterations, converged)``.
    """
    ln_chi_0 = math.log(chi_0)
    t_n = t_initial
    p = p_initial
    prev_delta_t: float | None = None

    for iteration in range(1, max_iter + 1):
        chi_hat = _compute_acceptance_ratio(e_min, e_max, t_n)

        if abs(chi_hat - chi_0) <= epsilon:
            return t_n, chi_hat, iteration, True

        # Guard against log(0) or log(1) — chi_hat outside (0, 1) exclusive.
        if chi_hat <= 0.0 or chi_hat >= 1.0:
            return t_n, chi_hat, iteration, False

        ln_chi_hat = math.log(chi_hat)
        t_next = t_n * (ln_chi_hat / ln_chi_0) ** p

        # Oscillation detection: sign change in consecutive temperature deltas.
        current_delta_t = t_next - t_n
        if prev_delta_t is not None and current_delta_t * prev_delta_t < 0:
            p *= 2

        prev_delta_t = current_delta_t
        t_n = t_next

    # Exhausted iterations — return best estimate.
    chi_hat = _compute_acceptance_ratio(e_min, e_max, t_n)
    return t_n, chi_hat, max_iter, abs(chi_hat - chi_0) <= epsilon


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_initial_temperature(
    instance: ProblemInstance,
    formulation: Literal["tqudo", "qubo"] = "tqudo",
    restriction: RestrictionConfig | None = None,
    *,
    chi_0: float = 0.8,
    n_samples: int = 200,
    epsilon: float = 1e-3,
    p_initial: float = 1.0,
    max_iter: int = 50,
    seed: int | None = None,
) -> T0EstimationResult:
    """Estimate the SA initial temperature for a target acceptance ratio.

    Implements the iterative procedure of Ben-Ameur (2004):

    1. Sample *n_samples* strictly-uphill transitions to collect
       ``(E_min, E_max)`` pairs.
    2. Compute an initial trial temperature from the mean cost increment.
    3. Iteratively refine until the estimated acceptance ratio matches *chi_0*
       within *epsilon*.

    Args:
        instance: Problem instance (small instances recommended for speed).
        formulation: ``"tqudo"`` or ``"qubo"``.
        restriction: Penalty weights; uses defaults if *None*.
        chi_0: Target acceptance ratio in ``(0, 1)``.
        n_samples: Number of uphill transitions to collect.
        epsilon: Convergence tolerance on ``|χ̂ - χ₀|``.
        p_initial: Initial exponent for the temperature update rule.
        max_iter: Maximum refinement iterations.
        seed: Optional RNG seed for reproducibility.

    Returns:
        :class:`T0EstimationResult` with the estimated temperature and
        convergence diagnostics.

    Raises:
        ValueError: If *formulation* is not ``"tqudo"`` or ``"qubo"``,
            or if *chi_0* is not in ``(0, 1)``.

    """
    if formulation not in ("tqudo", "qubo"):
        raise ValueError(f"formulation must be 'tqudo' or 'qubo', got {formulation!r}")
    if not (0.0 < chi_0 < 1.0):
        raise ValueError(f"chi_0 must be in (0, 1), got {chi_0}")

    restriction = restriction or _default_restriction()
    rng = np.random.default_rng(seed)
    n_available = instance.n_cities - 1

    # Build formulation tensors / matrix.
    if formulation == "tqudo":
        problem: ProblemQUBO | ProblemTQUDO = generate_TQUDO_from_problem(
            instance,
            restriction,
        )
    else:
        problem = generate_QUBO_from_problem(instance, restriction)

    # Step 1: collect uphill transitions.
    e_min, e_max = _collect_uphill_transitions(
        formulation,
        problem,
        n_available,
        n_samples,
        rng,
    )

    if len(e_min) == 0:
        # Degenerate landscape — no uphill transitions found.
        return T0EstimationResult(
            t0=_DEFAULT_FALLBACK_T0,
            chi_achieved=1.0,
            iterations=0,
            n_samples=0,
            converged=False,
        )

    # Initial trial temperature: T_1 = -mean(δ) / ln(χ₀).
    deltas = e_max - e_min
    mean_delta = float(np.mean(deltas))
    t_1 = -mean_delta / math.log(chi_0)

    # Step 2: iterative refinement.
    t0, chi_achieved, iterations, converged = _refine_temperature(
        e_min,
        e_max,
        chi_0,
        t_1,
        epsilon,
        p_initial,
        max_iter,
    )

    return T0EstimationResult(
        t0=t0,
        chi_achieved=chi_achieved,
        iterations=iterations,
        n_samples=len(e_min),
        converged=converged,
    )
