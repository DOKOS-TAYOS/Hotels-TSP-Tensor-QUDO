"""Pairwise merge and dashboard row statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from data_analysis._deps import coerce_bool_scalar
from data_analysis.benchmark.common import _ATOL_REAL, _RTOL_REAL


def is_optimal_vs_ref(
    real_cost: float | None,
    ref_real_cost: float | None,
    feasible: Any,
) -> bool:
    if ref_real_cost is None or (isinstance(ref_real_cost, float) and math.isnan(ref_real_cost)):
        return False
    if real_cost is None or (isinstance(real_cost, float) and math.isnan(real_cost)):
        return False
    if not coerce_bool_scalar(feasible):
        return False
    return bool(
        math.isclose(
            float(real_cost),
            float(ref_real_cost),
            rel_tol=_RTOL_REAL,
            abs_tol=_ATOL_REAL,
        )
    )


def _dedupe_solution_rows(df: Any, keys: list[str]) -> Any:
    if df.empty:
        return df
    out = df.copy()
    if "path" in out.columns:
        out = out.sort_values("path")
    return out.groupby(keys, as_index=False, sort=False).last()


def _merge_paired(
    paired: Any,
    *,
    left: tuple[str, str],
    right: tuple[str, str],
    dedupe_keys: list[str],
    merge_on: list[str],
    n_cities_filter: int | None = None,
) -> Any:
    s_a, f_a = left
    s_b, f_b = right

    def _take(solver: str, form: str) -> Any:
        m = (
            paired["parse_ok"]
            & paired["solve_ok"]
            & (paired["solver"] == solver)
            & (paired["formulation"] == form)
        )
        if n_cities_filter is not None:
            m = m & (paired["n_cities"] == n_cities_filter)
        return paired[m].copy()

    da = _dedupe_solution_rows(_take(s_a, f_a), dedupe_keys)
    db = _dedupe_solution_rows(_take(s_b, f_b), dedupe_keys)
    return da.merge(db, on=merge_on, how="inner", suffixes=("_left", "_right"))


def _stats_dashboard_left_only_from_runs(runs: Any) -> dict[str, float | int]:
    """Same keys as :func:`_stats_from_rows`, but for a single-solver cohort (left bars only).

    Right-side counts are zero; conditional cost / ``only_*`` totals are cleared so the
    row is safe for the stacked-feasibility panel when the opposing solver has no data at
    that depth (e.g. CUDA-Q ``tqudo_virtual`` alone at :math:`p=3`, :math:`n=9`).
    """
    n = int(len(runs))
    out: dict[str, float | int] = {
        "n_paired": n,
        "left_optimal": 0,
        "left_feasible_subopt": 0,
        "left_infeasible": 0,
        "right_optimal": 0,
        "right_feasible_subopt": 0,
        "right_infeasible": 0,
        "n_both_feasible": 0,
        "cost_left_better_cond": 0,
        "cost_right_better_cond": 0,
        "cost_tie_cond": 0,
        "cost_left_better_cond_pct": float("nan"),
        "cost_right_better_cond_pct": float("nan"),
        "cost_tie_cond_pct": float("nan"),
        "only_left_feasible": 0,
        "only_right_feasible": 0,
        "only_left_optimal": 0,
        "only_right_optimal": 0,
        "only_left_feasible_pct": float("nan"),
        "only_right_feasible_pct": float("nan"),
        "only_left_optimal_pct": float("nan"),
        "only_right_optimal_pct": float("nan"),
    }
    if n == 0:
        return out

    l_feas = runs["feasible"].map(coerce_bool_scalar)
    l_opt = runs.apply(
        lambda r: is_optimal_vs_ref(r["real_cost"], r["ref_real_cost"], r["feasible"]),
        axis=1,
    )
    out["left_optimal"] = int(l_opt.sum())
    out["left_feasible_subopt"] = int((l_feas & ~l_opt).sum())
    out["left_infeasible"] = int((~l_feas).sum())
    return out


def _stats_from_rows(merged: Any) -> dict[str, float | int]:
    """Compute dashboard metrics on inner-joined paired rows (suffixes _left / _right)."""
    n = int(len(merged))
    out: dict[str, float | int] = {
        "n_paired": n,
        "left_optimal": 0,
        "left_feasible_subopt": 0,
        "left_infeasible": 0,
        "right_optimal": 0,
        "right_feasible_subopt": 0,
        "right_infeasible": 0,
        "n_both_feasible": 0,
        "cost_left_better_cond": 0,
        "cost_right_better_cond": 0,
        "cost_tie_cond": 0,
        "cost_left_better_cond_pct": float("nan"),
        "cost_right_better_cond_pct": float("nan"),
        "cost_tie_cond_pct": float("nan"),
        "only_left_feasible": 0,
        "only_right_feasible": 0,
        "only_left_optimal": 0,
        "only_right_optimal": 0,
        "only_left_feasible_pct": float("nan"),
        "only_right_feasible_pct": float("nan"),
        "only_left_optimal_pct": float("nan"),
        "only_right_optimal_pct": float("nan"),
    }
    if n == 0:
        return out

    l_feas = merged["feasible_left"].map(coerce_bool_scalar)
    r_feas = merged["feasible_right"].map(coerce_bool_scalar)
    l_rc = merged["real_cost_left"]
    r_rc = merged["real_cost_right"]

    l_opt = merged.apply(
        lambda r: is_optimal_vs_ref(
            r["real_cost_left"], r["ref_real_cost_left"], r["feasible_left"]
        ),
        axis=1,
    )
    r_opt = merged.apply(
        lambda r: is_optimal_vs_ref(
            r["real_cost_right"], r["ref_real_cost_right"], r["feasible_right"]
        ),
        axis=1,
    )

    out["left_optimal"] = int(l_opt.sum())
    out["left_feasible_subopt"] = int((l_feas & ~l_opt).sum())
    out["left_infeasible"] = int((~l_feas).sum())
    out["right_optimal"] = int(r_opt.sum())
    out["right_feasible_subopt"] = int((r_feas & ~r_opt).sum())
    out["right_infeasible"] = int((~r_feas).sum())

    out["only_left_feasible"] = int((l_feas & ~r_feas).sum())
    out["only_right_feasible"] = int((r_feas & ~l_feas).sum())
    out["only_left_optimal"] = int((l_opt & ~r_opt).sum())
    out["only_right_optimal"] = int((r_opt & ~l_opt).sum())
    out["only_left_feasible_pct"] = 100.0 * float(out["only_left_feasible"]) / float(n)
    out["only_right_feasible_pct"] = 100.0 * float(out["only_right_feasible"]) / float(n)
    out["only_left_optimal_pct"] = 100.0 * float(out["only_left_optimal"]) / float(n)
    out["only_right_optimal_pct"] = 100.0 * float(out["only_right_optimal"]) / float(n)

    both_feas = l_feas & r_feas & l_rc.notna() & r_rc.notna()
    n_both = int(both_feas.sum())
    out["n_both_feasible"] = n_both
    if n_both == 0:
        return out

    sub = merged.loc[both_feas]
    a = sub["real_cost_left"].astype(float).to_numpy()
    c = sub["real_cost_right"].astype(float).to_numpy()
    tie = np.array(
        [
            math.isclose(float(a[i]), float(c[i]), rel_tol=_RTOL_REAL, abs_tol=_ATOL_REAL)
            for i in range(len(sub))
        ],
        dtype=bool,
    )
    left_lower = (~tie) & (a < c)
    right_lower = (~tie) & (a > c)
    out["cost_left_better_cond"] = int(left_lower.sum())
    out["cost_right_better_cond"] = int(right_lower.sum())
    out["cost_tie_cond"] = int(tie.sum())
    nn = float(n_both)
    out["cost_left_better_cond_pct"] = 100.0 * float(out["cost_left_better_cond"]) / nn
    out["cost_right_better_cond_pct"] = 100.0 * float(out["cost_right_better_cond"]) / nn
    out["cost_tie_cond_pct"] = 100.0 * float(out["cost_tie_cond"]) / nn

    return out
