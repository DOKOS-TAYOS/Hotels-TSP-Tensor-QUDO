"""Paired comparative dashboards and mean approximation-ratio plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from data_analysis._deps import coerce_bool_scalar
from data_analysis._plot_typography import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_FONTSIZE_COMPACT,
    TICK_LABEL_FONTSIZE,
)
from data_analysis.metrics import (
    first_optimizer_step_reaching_min_energy,
    read_energy_history_from_solution_json,
)
from data_analysis.optimal_sample_mass import (
    histogram_key_for_formulation,
    histogram_mass,
    load_bruteforce_optimal_sequence,
    read_sample_histograms_from_solution_json,
)

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


def _clip_values_for_log_y(vals: list[float], *, floor: float = _P_OPT_LOG_AXIS_FLOOR) -> list[float]:
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


def _clip_mean_std_for_log_y(
    means: list[float],
    stds: list[float],
    *,
    floor: float = _LOG_Y_FLOOR,
) -> tuple[list[float], list[float]]:
    """Clip means to ``floor`` and cap symmetric error so ``mean - std >= floor`` (for log-scaled *y*)."""
    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(stds, dtype=np.float64)
    m = np.where(np.isfinite(m), m, np.nan)
    s = np.where(np.isfinite(s), s, 0.0)
    m_plot = np.maximum(m, floor)
    s_cap = np.minimum(s, np.maximum(m_plot - floor, 0.0))
    return m_plot.tolist(), s_cap.tolist()


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
        "cost_left_better_cond_pct": float("nan"),
        "cost_right_better_cond_pct": float("nan"),
        "cost_tie_cond_pct": float("nan"),
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
        lambda r: is_optimal_vs_ref(r["real_cost_left"], r["ref_real_cost_left"], r["feasible_left"]),
        axis=1,
    )
    r_opt = merged.apply(
        lambda r: is_optimal_vs_ref(r["real_cost_right"], r["ref_real_cost_right"], r["feasible_right"]),
        axis=1,
    )

    out["left_optimal"] = int(l_opt.sum())
    out["left_feasible_subopt"] = int((l_feas & ~l_opt).sum())
    out["left_infeasible"] = int((~l_feas).sum())
    out["right_optimal"] = int(r_opt.sum())
    out["right_feasible_subopt"] = int((r_feas & ~r_opt).sum())
    out["right_infeasible"] = int((~r_feas).sum())

    out["only_left_feasible_pct"] = 100.0 * float((l_feas & ~r_feas).sum()) / float(n)
    out["only_right_feasible_pct"] = 100.0 * float((r_feas & ~l_feas).sum()) / float(n)
    out["only_left_optimal_pct"] = 100.0 * float((l_opt & ~r_opt).sum()) / float(n)
    out["only_right_optimal_pct"] = 100.0 * float((r_opt & ~l_opt).sum()) / float(n)

    both_feas = l_feas & r_feas & l_rc.notna() & r_rc.notna()
    n_both = int(both_feas.sum())
    if n_both == 0:
        return out

    sub = merged.loc[both_feas]
    a = sub["real_cost_left"].astype(float).to_numpy()
    c = sub["real_cost_right"].astype(float).to_numpy()
    tie = np.array(
        [math.isclose(float(a[i]), float(c[i]), rel_tol=_RTOL_REAL, abs_tol=_ATOL_REAL) for i in range(len(sub))],
        dtype=bool,
    )
    left_lower = (~tie) & (a < c)
    right_lower = (~tie) & (a > c)
    nn = float(n_both)
    out["cost_left_better_cond_pct"] = 100.0 * float(left_lower.sum()) / nn
    out["cost_right_better_cond_pct"] = 100.0 * float(right_lower.sum()) / nn
    out["cost_tie_cond_pct"] = 100.0 * float(tie.sum()) / nn

    return out


def _plot_comparison_dashboard(
    *,
    x_labels: list[str],
    stats_list: list[dict[str, float | int]],
    label_left: str,
    label_right: str,
    x_axis_label: str,
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    x = np.arange(len(x_labels), dtype=np.float64)
    w = 0.36
    c_opt, c_sub, c_inf = "#2ca02c", "#ffbb78", "#c7c7c7"

    ax00 = axes[0, 0]
    left_opt = [float(s["left_optimal"]) for s in stats_list]
    left_sub = [float(s["left_feasible_subopt"]) for s in stats_list]
    left_inf = [float(s["left_infeasible"]) for s in stats_list]
    right_opt = [float(s["right_optimal"]) for s in stats_list]
    right_sub = [float(s["right_feasible_subopt"]) for s in stats_list]
    right_inf = [float(s["right_infeasible"]) for s in stats_list]

    ax00.bar(
        x - w / 2, left_opt, w, label="Optimal", color=c_opt, edgecolor="white", linewidth=0.5
    )
    ax00.bar(
        x - w / 2,
        left_sub,
        w,
        bottom=left_opt,
        label="Suboptimal",
        color=c_sub,
        edgecolor="white",
        linewidth=0.5,
    )
    ax00.bar(
        x - w / 2,
        left_inf,
        w,
        bottom=np.array(left_opt) + np.array(left_sub),
        label="Infeasible",
        color=c_inf,
        edgecolor="white",
        linewidth=0.5,
    )

    ax00.bar(x + w / 2, right_opt, w, color=c_opt, edgecolor="white", linewidth=0.5)
    ax00.bar(
        x + w / 2,
        right_sub,
        w,
        bottom=right_opt,
        color=c_sub,
        edgecolor="white",
        linewidth=0.5,
    )
    ax00.bar(
        x + w / 2,
        right_inf,
        w,
        bottom=np.array(right_opt) + np.array(right_sub),
        color=c_inf,
        edgecolor="white",
        linewidth=0.5,
    )

    ax00.set_xticks(x)
    ax00.set_xticklabels(x_labels)
    ax00.set_ylabel("Instances", fontsize=AXIS_LABEL_FONTSIZE)
    ax00.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    ax00.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax00.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    ax01 = axes[0, 1]
    cl = [float(s.get("cost_left_better_cond_pct", np.nan)) for s in stats_list]
    cr = [float(s.get("cost_right_better_cond_pct", np.nan)) for s in stats_list]
    ct = [float(s.get("cost_tie_cond_pct", np.nan)) for s in stats_list]
    ww = 0.25
    ax01.bar(x - ww, cl, ww, label=f"Lower cost ({label_left})", color="#1f77b4")
    ax01.bar(x, cr, ww, label=f"Lower cost ({label_right})", color="#ff7f0e")
    ax01.bar(x + ww, ct, ww, label="Tie", color="#7f7f7f")
    ax01.set_xticks(x)
    ax01.set_xticklabels(x_labels)
    ax01.set_ylabel(
        "% lower real cost\n(both feasible)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax01.set_ylim(0, 105)
    ax01.legend(fontsize=LEGEND_FONTSIZE_COMPACT)
    ax01.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax01.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    ax10 = axes[1, 0]
    olf = [float(s["only_left_feasible_pct"]) for s in stats_list]
    orf = [float(s["only_right_feasible_pct"]) for s in stats_list]
    ax10.bar(x - ww, olf, ww, label=f"{label_left} only", color="#1f77b4")
    ax10.bar(x + ww, orf, ww, label=f"{label_right} only", color="#ff7f0e")
    ax10.set_xticks(x)
    ax10.set_xticklabels(x_labels)
    ax10.set_ylabel(
        "% feasible\n(one side only)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax10.set_ylim(0, 105)
    ax10.legend(fontsize=LEGEND_FONTSIZE)
    ax10.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax10.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    ax11 = axes[1, 1]
    olo = [float(s["only_left_optimal_pct"]) for s in stats_list]
    oro = [float(s["only_right_optimal_pct"]) for s in stats_list]
    ax11.bar(x - ww, olo, ww, label=f"{label_left} only", color="#2ca02c")
    ax11.bar(x + ww, oro, ww, label=f"{label_right} only", color="#d62728")
    ax11.set_xticks(x)
    ax11.set_xticklabels(x_labels)
    ax11.set_ylabel(
        "% optimal\n(one side only)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax11.set_ylim(0, 105)
    ax11.legend(fontsize=LEGEND_FONTSIZE)
    ax11.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax11.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout()
    return fig


def _opt_steps_from_rel_path(output_root: Path, rel_path: Any) -> float | None:
    """1-based step count to first trace minimum from JSON at ``output_root / rel_path``."""
    if rel_path is None:
        return None
    s = str(rel_path).strip()
    if not s or s.lower() == "nan":
        return None
    h = read_energy_history_from_solution_json(output_root / s)
    if not h:
        return None
    step = first_optimizer_step_reaching_min_energy(h)
    return float(step) if step is not None else None


def _collect_side_opt_step_lists_by_depth(
    merged: Any,
    *,
    depths: tuple[int, ...],
    output_root: Path,
) -> tuple[list[list[float]], list[list[float]]]:
    """Per QAOA depth: raw step counts left/right (paired rows optimal on that side only)."""
    empty = ([[] for _ in depths], [[] for _ in depths])
    if merged.empty or "path_left" not in merged.columns:
        return empty
    lists_l: list[list[float]] = []
    lists_r: list[list[float]] = []
    for d in depths:
        sub = merged[_mask_qaoa_depth_eq(merged["qaoa_depth"], int(d))]
        steps_l: list[float] = []
        steps_r: list[float] = []
        for _, row in sub.iterrows():
            if is_optimal_vs_ref(
                row["real_cost_left"], row["ref_real_cost_left"], row["feasible_left"]
            ):
                sl = _opt_steps_from_rel_path(output_root, row.get("path_left"))
                if sl is not None:
                    steps_l.append(float(sl))
            if is_optimal_vs_ref(
                row["real_cost_right"], row["ref_real_cost_right"], row["feasible_right"]
            ):
                sr = _opt_steps_from_rel_path(output_root, row.get("path_right"))
                if sr is not None:
                    steps_r.append(float(sr))
        lists_l.append(steps_l)
        lists_r.append(steps_r)
    return lists_l, lists_r


def _step_lists_to_depth_dict(
    depths: tuple[int, ...], lists: list[list[float]]
) -> dict[int, list[float]]:
    return {int(d): list(vals) for d, vals in zip(depths, lists, strict=True)}


def _opt_steps_values_cell(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
    qaoa_depth: int,
    output_root: Path,
) -> list[float]:
    """Step counts to first trace minimum for runs optimal vs BF ref at one (n, p)."""
    qd = paired["qaoa_depth"]
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == n_cities)
        & _mask_qaoa_depth_eq(qd, int(qaoa_depth))
    )
    sub = paired.loc[m].copy()
    if sub.empty:
        return []
    sub = _dedupe_solution_rows(
        sub,
        ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
    )
    steps: list[float] = []
    for _, row in sub.iterrows():
        if not is_optimal_vs_ref(row["real_cost"], row["ref_real_cost"], row["feasible"]):
            continue
        st = _opt_steps_from_rel_path(output_root, row.get("path"))
        if st is not None and np.isfinite(float(st)):
            steps.append(float(st))
    return steps


def _collect_cirq_tqudo_opt_steps_box_series_vs_ncities(
    paired: Any,
    *,
    n_values: list[int],
    depth_values: tuple[int, ...],
    output_root: Path,
) -> list[tuple[str, list[float], list[list[float]]]]:
    """Per QAOA depth: dodged *n*, raw step counts for Cirq native TQUDO (optimal runs only)."""
    depth_union = set(depth_values)
    if not depth_union:
        return []
    depths_sorted = sorted(depth_union)
    dodge_step = 0.14
    half = 0.5 * float(len(depths_sorted) - 1) if len(depths_sorted) > 1 else 0.0
    out: list[tuple[str, list[float], list[list[float]]]] = []
    for rank, depth in enumerate(depths_sorted):
        x_off = (float(rank) - half) * dodge_step if len(depths_sorted) > 1 else 0.0
        xs: list[float] = []
        datas: list[list[float]] = []
        for n in n_values:
            vals = _opt_steps_values_cell(
                paired,
                solver="cirq",
                formulation="tqudo",
                n_cities=n,
                qaoa_depth=depth,
                output_root=output_root,
            )
            if not vals:
                continue
            xs.append(float(n) + x_off)
            datas.append(vals)
        if xs:
            out.append((f"p = {depth}", xs, datas))
    return out


def _mean_approx_ratio_by_depth_unpaired(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
) -> tuple[list[int], list[float], list[float]]:
    """Per QAOA depth: mean and sample std of ``approx_ratio_real`` (feasible rows only).

    Not paired across formulations. One row per (instance_key, depth) after dedupe.
    Depths with no qualifying rows omitted. For a single sample, std is 0.
    """
    if "approx_ratio_real" not in paired.columns:
        return [], [], []

    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == n_cities)
        & paired["qaoa_depth"].notna()
    )
    sub = paired.loc[m].copy()
    if sub.empty:
        return [], [], []
    sub = _dedupe_solution_rows(
        sub,
        ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
    )
    feas = sub["feasible"].map(coerce_bool_scalar)
    ar = sub["approx_ratio_real"]
    sub = sub.loc[feas & ar.notna()].copy()
    if sub.empty:
        return [], [], []
    sub["qaoa_depth"] = sub["qaoa_depth"].astype(int)
    grouped = sub.groupby("qaoa_depth", sort=True)["approx_ratio_real"].agg(["mean", "std"])
    depths = [int(d) for d in grouped.index.tolist()]
    means = [float(x) for x in grouped["mean"].to_numpy()]
    stds = [float(x) for x in grouped["std"].fillna(0.0).to_numpy()]
    return depths, means, stds


def _approx_ratio_lists_by_depth_unpaired(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
) -> dict[int, list[float]]:
    """Per QAOA depth: list of ``approx_ratio_real`` (feasible rows only, deduped).

    Same filtering as ``_mean_approx_ratio_by_depth_unpaired``; used for boxplots.
    """
    if "approx_ratio_real" not in paired.columns:
        return {}

    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == n_cities)
        & paired["qaoa_depth"].notna()
    )
    sub = paired.loc[m].copy()
    if sub.empty:
        return {}
    sub = _dedupe_solution_rows(
        sub,
        ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
    )
    feas = sub["feasible"].map(coerce_bool_scalar)
    ar = sub["approx_ratio_real"]
    sub = sub.loc[feas & ar.notna()].copy()
    if sub.empty:
        return {}
    sub["qaoa_depth"] = sub["qaoa_depth"].astype(int)
    out: dict[int, list[float]] = {}
    for depth, grp in sub.groupby("qaoa_depth", sort=True):
        vals = [float(v) for v in grp["approx_ratio_real"].to_numpy() if np.isfinite(v)]
        if vals:
            out[int(depth)] = vals
    return out


def _solver_form_tqudo_by_n_cities(n_cities: int) -> tuple[str, str]:
    """TQUDO qudits (Cirq native) for n<9; TQUDO qubits (CUDA-Q) for n=9 (project convention)."""
    if n_cities == 9:
        return ("cudaq", "tqudo_virtual")
    return ("cirq", "tqudo")


def _mean_approx_ratio_cell(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
    qaoa_depth: int,
) -> tuple[float, float] | None:
    """Mean and sample std at one (n_cities, QAOA depth); ``None`` if no feasible rows."""
    if "approx_ratio_real" not in paired.columns:
        return None
    qd = paired["qaoa_depth"]
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == n_cities)
        & qd.notna()
        & (qd.astype(float) == float(qaoa_depth))
    )
    sub = paired.loc[m].copy()
    if sub.empty:
        return None
    sub = _dedupe_solution_rows(
        sub,
        ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
    )
    feas = sub["feasible"].map(coerce_bool_scalar)
    ar = sub["approx_ratio_real"]
    sub = sub.loc[feas & ar.notna(), "approx_ratio_real"]
    if sub.empty:
        return None
    mean = float(sub.mean())
    std = float(sub.std(ddof=1)) if len(sub) > 1 else 0.0
    if std != std:  # NaN
        std = 0.0
    return mean, std


def _approx_ratio_values_cell(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
    qaoa_depth: int,
) -> list[float]:
    """All ``approx_ratio_real`` at one (``n_cities``, QAOA depth); empty if none."""
    if "approx_ratio_real" not in paired.columns:
        return []
    qd = paired["qaoa_depth"]
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == n_cities)
        & qd.notna()
        & (qd.astype(float) == float(qaoa_depth))
    )
    sub = paired.loc[m].copy()
    if sub.empty:
        return []
    sub = _dedupe_solution_rows(
        sub,
        ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
    )
    feas = sub["feasible"].map(coerce_bool_scalar)
    ar = sub["approx_ratio_real"]
    sub = sub.loc[feas & ar.notna(), "approx_ratio_real"]
    return [float(v) for v in sub.to_numpy() if np.isfinite(v)]


def _mean_approx_ratio_series_vs_ncities_by_depth(
    paired: Any,
    *,
    n_values: list[int],
) -> list[tuple[str, list[float], list[float], list[float]]]:
    """One errorbar series per QAOA depth: x = n_cities (dodged), y = mean ratio, yerr = std."""
    depth_union: set[int] = set()
    for n in n_values:
        s, f = _solver_form_tqudo_by_n_cities(n)
        dlist, _, _ = _mean_approx_ratio_by_depth_unpaired(
            paired, solver=s, formulation=f, n_cities=n
        )
        depth_union.update(dlist)
    if not depth_union:
        return []

    depths_sorted = sorted(depth_union)
    dodge_step = 0.14
    half = 0.5 * float(len(depths_sorted) - 1) if len(depths_sorted) > 1 else 0.0
    out: list[tuple[str, list[float], list[float], list[float]]] = []
    for rank, depth in enumerate(depths_sorted):
        x_off = (float(rank) - half) * dodge_step if len(depths_sorted) > 1 else 0.0
        xs: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        for n in n_values:
            s, f = _solver_form_tqudo_by_n_cities(n)
            cell = _mean_approx_ratio_cell(
                paired,
                solver=s,
                formulation=f,
                n_cities=n,
                qaoa_depth=depth,
            )
            if cell is None:
                continue
            m, sd = cell
            xs.append(float(n) + x_off)
            means.append(m)
            stds.append(sd)
        if xs:
            out.append((f"p = {depth}", xs, means, stds))
    return out


def _approx_ratio_box_series_vs_ncities_by_depth(
    paired: Any,
    *,
    n_values: list[int],
) -> list[tuple[str, list[float], list[list[float]]]]:
    """One boxplot series per QAOA depth: x = n_cities (dodged), raw ρ lists per *n*."""
    depth_union: set[int] = set()
    for n in n_values:
        s, f = _solver_form_tqudo_by_n_cities(n)
        depth_union.update(
            _approx_ratio_lists_by_depth_unpaired(
                paired, solver=s, formulation=f, n_cities=n
            ).keys()
        )
    if not depth_union:
        return []

    depths_sorted = sorted(depth_union)
    dodge_step = 0.14
    half = 0.5 * float(len(depths_sorted) - 1) if len(depths_sorted) > 1 else 0.0
    out: list[tuple[str, list[float], list[list[float]]]] = []
    for rank, depth in enumerate(depths_sorted):
        x_off = (float(rank) - half) * dodge_step if len(depths_sorted) > 1 else 0.0
        xs: list[float] = []
        datas: list[list[float]] = []
        for n in n_values:
            s, f = _solver_form_tqudo_by_n_cities(n)
            vals = _approx_ratio_values_cell(
                paired,
                solver=s,
                formulation=f,
                n_cities=n,
                qaoa_depth=depth,
            )
            if not vals:
                continue
            xs.append(float(n) + x_off)
            datas.append(vals)
        if xs:
            out.append((f"p = {depth}", xs, datas))
    return out


def _plot_mean_approx_ratio_vs_ncities(
    series: list[tuple[str, list[float], list[float], list[float]]],
    *,
    n_tick_vals: list[int],
    figsize: tuple[float, float] = (8.0, 5.0),
    y_label: str | None = None,
    ref_hline: float | None = 1.0,
    ref_hline_label: str | None = None,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
) -> Any:
    """``series``: (label, x, y, yerr) with x already including depth dodge.

    ``y_scale``: ``"linear"`` (default), ``"log"`` (positive *y*; clips to small floor), or
    ``"symlog"`` for signed values (e.g. :math:`\\Delta P`).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    markers = ("o", "s", "^", "D", "v", "P", "*", "X", "h", "8")
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    for i, (label, xs, means, stds) in enumerate(series):
        if not xs:
            continue
        if y_scale == "log":
            ym, es = _clip_mean_std_for_log_y(means, stds)
            ax.errorbar(
                xs,
                ym,
                yerr=es,
                fmt=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle="None",
                markersize=8,
                capsize=4,
                elinewidth=1.2,
                label=label,
            )
        else:
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                fmt=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle="None",
                markersize=8,
                capsize=4,
                elinewidth=1.2,
                label=label,
            )
    if ref_hline is not None and math.isfinite(float(ref_hline)):
        if y_scale != "log" or float(ref_hline) > 0.0:
            rlab = ref_hline_label if ref_hline_label is not None else r"$\rho = 1$"
            ax.axhline(
                float(ref_hline), color="gray", linestyle="--", linewidth=1, label=rlab
            )
    ax.set_xticks([float(n) for n in n_tick_vals])
    ax.set_xticklabels([str(n) for n in n_tick_vals])
    ax.set_xlabel(r"$n$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        y_label if y_label is not None else r"$\rho$ (mean ± σ)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    if y_scale == "log":
        ax.set_yscale("log")
    elif y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh, base=10)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(
        fontsize=LEGEND_FONTSIZE_COMPACT if len(series) > 4 else LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    return fig


def _plot_mean_approx_ratio_points(
    series: list[tuple[str, list[int], list[float], list[float]]],
    *,
    figsize: tuple[float, float] = (7.5, 4.8),
    y_label: str | None = None,
    ref_hline: float | None = 1.0,
    ref_hline_label: str | None = None,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    markers = ("o", "s", "^", "D", "v", "P", "*", "X", "h", "8")
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    active = [(i, t) for i, t in enumerate(series) if t[1]]
    n_active = len(active)
    dodge_step = 0.09
    half = 0.5 * float(n_active - 1) if n_active else 0.0
    for rank, (i, (label, depths, means, stds)) in enumerate(active):
        x_off = (float(rank) - half) * dodge_step if n_active > 1 else 0.0
        xs = [int(d) + x_off for d in depths]
        if y_scale == "log":
            ym, es = _clip_mean_std_for_log_y(means, stds)
            ax.errorbar(
                xs,
                ym,
                yerr=es,
                fmt=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle="None",
                markersize=8,
                capsize=4,
                elinewidth=1.2,
                label=label,
            )
        else:
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                fmt=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle="None",
                markersize=8,
                capsize=4,
                elinewidth=1.2,
                label=label,
            )
    if ref_hline is not None and math.isfinite(float(ref_hline)):
        if y_scale != "log" or float(ref_hline) > 0.0:
            rlab = ref_hline_label if ref_hline_label is not None else r"$\rho = 1$"
            ax.axhline(
                float(ref_hline), color="gray", linestyle="--", linewidth=1, label=rlab
            )
    all_depths = sorted({d for _, ds, _, _ in series for d in ds})
    if all_depths:
        ax.set_xticks(all_depths)
    ax.set_xlabel(r"$p$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        y_label if y_label is not None else r"$\rho$ (mean ± σ)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    if y_scale == "log":
        ax.set_yscale("log")
    elif y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh, base=10)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(
        fontsize=LEGEND_FONTSIZE_COMPACT if len(series) > 4 else LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    return fig


def _scatter_rho_instances_jittered(
    ax: Any,
    x_center: float,
    values: list[float],
    *,
    color: str,
    jitter_span: float,
    rng: np.random.Generator,
) -> None:
    """Jittered semi-transparent markers so stacked :math:`\\rho=1` instances remain visible."""
    if not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    j = (rng.random(arr.size) - 0.5) * jitter_span
    ax.scatter(
        x_center + j,
        arr,
        s=18,
        c=color,
        alpha=0.32,
        edgecolors="white",
        linewidths=0.25,
        zorder=4,
    )


def _ylim_rho_plot(all_values: list[float], *, y_scale: str) -> tuple[float, float]:
    if not all_values:
        return (1.0, 1.05) if y_scale == "log" else (1.0, 1.05)
    arr = np.asarray(all_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (1.0, 1.05) if y_scale == "log" else (1.0, 1.05)
    mx = float(np.max(arr))
    if y_scale == "log":
        return 1.0, max(mx * 1.12, 1.02)
    mn = float(np.min(arr))
    bottom = 1.0 if mn >= 1.0 else max(0.95, mn * 0.98)
    pad = max((mx - 1.0) * 0.1, 0.04) if mx > 1.0 else 0.05
    return bottom, max(mx + pad, bottom + 0.04)


def _decorate_approx_ratio_y_axis(
    ax: Any,
    lo: float,
    hi: float,
    *,
    y_scale: str,
    symlog_linthresh: float,
) -> None:
    """Readable ticks for :math:`\\rho \\gtrsim 1` (linear default; log uses per-decade multipliers)."""
    from matplotlib.ticker import LogLocator, MaxNLocator, MultipleLocator, ScalarFormatter

    if y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh, base=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    elif y_scale == "log":
        ax.set_yscale("log")
        subs = tuple(float(x) for x in range(1, 10))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs, numticks=15))
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        fmt.set_useOffset(False)
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.set_yscale("linear")
        span = max(hi - lo, 1e-9)
        if span <= 0.2:
            step = 0.02 if span <= 0.1 else 0.05
            ax.yaxis.set_major_locator(MultipleLocator(step))
        elif span <= 0.5:
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
        else:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10], min_n_ticks=5)
            )
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)

    ax.yaxis.grid(True, which="major", linestyle=":", alpha=0.35)
    if y_scale == "log":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.15)


def _decorate_y_axis_from_values(
    ax: Any,
    all_y: list[float],
    *,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    asinh_linear_width: float | None = None,
    log_y_clip_upper: float | None = None,
    manual_y_limits: bool = True,
) -> None:
    """Y limits with padding and readable ticks for generic numeric samples.

    ``y_scale="asinh"`` uses Matplotlib's inverse-sinh axis (often clearer ticks
    than ``symlog`` for signed values). When ``asinh_linear_width`` is omitted,
    ``symlog_linthresh`` is used as the ``linear_width`` parameter.

    For ``y_scale="log"``, if ``log_y_clip_upper`` is set (e.g. ``1.0`` for
    :math:`P(\\mathrm{opt})`), the upper *y* limit is capped at that value.

    If ``manual_y_limits`` is false, no ``set_ylim`` is applied; the axis uses
    Matplotlib autoscaling from artists already drawn on *ax*.
    """
    from matplotlib.ticker import (
        AsinhLocator,
        LogFormatterSciNotation,
        LogLocator,
        MaxNLocator,
        ScalarFormatter,
        SymmetricalLogLocator,
    )

    arr = np.asarray([v for v in all_y if np.isfinite(v)], dtype=np.float64)
    if y_scale == "log":
        pos = arr[arr > 0]
        if pos.size == 0:
            lo, hi = float(_P_OPT_LOG_AXIS_FLOOR), 1.0
        else:
            lo_raw = float(np.min(pos))
            hi = float(np.max(arr))
            hi = max(hi, lo_raw)
            # Padding below the smallest *y*; do not clamp to ``_P_OPT_LOG_AXIS_FLOOR``
            # (too high—hides uniform references like ``1/2^{(n-1)^2}`` for large *n*).
            lo = max(lo_raw * 0.62, float(np.finfo(float).tiny))
            hi = max(hi * 1.18, lo_raw * 1.06)
            if log_y_clip_upper is not None:
                hi = min(hi, float(log_y_clip_upper))
        if lo >= hi:
            hi = lo * 10.0
            if log_y_clip_upper is not None:
                hi = min(hi, float(log_y_clip_upper))
        ax.set_yscale("log")
        if manual_y_limits:
            ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    else:
        if arr.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(np.min(arr)), float(np.max(arr))
            span = hi - lo
            abs_max = float(np.max(np.abs(arr)))
            edge_max = max(abs(lo), abs(hi))
            if span > 0:
                pad = max(span * 0.18, 0.03 * edge_max, 0.02 * abs_max, 1e-9)
            else:
                pad = max(0.05 * edge_max, 0.04 * abs_max, 0.02, 1e-9)
            lo, hi = lo - pad, hi + pad
        if y_scale == "asinh":
            lw = float(asinh_linear_width) if asinh_linear_width is not None else float(symlog_linthresh)
            lw = max(lw, 1e-15)
            ax.set_yscale("asinh", linear_width=lw)
            if manual_y_limits:
                ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(AsinhLocator(lw, numticks=8))
        elif y_scale == "symlog":
            absv = np.abs(arr[np.isfinite(arr)])
            absv_nz = absv[absv > 1e-15]
            if absv_nz.size:
                med = float(np.median(absv_nz))
                eff_lt = max(symlog_linthresh, min(med * 0.15, 0.05))
            else:
                eff_lt = max(symlog_linthresh, 1e-6)
            ax.set_yscale("symlog", linthresh=eff_lt, base=10)
            if manual_y_limits:
                ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=eff_lt, base=10))
        else:
            ax.set_yscale("linear")
            if manual_y_limits:
                ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    if not manual_y_limits:
        ax.relim(visible_only=True)
        ax.autoscale(axis="y")
    ax.yaxis.grid(True, which="major", linestyle=":", alpha=0.35)
    if y_scale == "log":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.2)
    elif y_scale == "symlog":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.22)
    elif y_scale == "asinh":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.18)


def _plot_approx_ratio_boxplots_vs_p(
    series: list[tuple[str, dict[int, list[float]]]],
    *,
    figsize: tuple[float, float] = (7.5, 4.8),
    y_label: str | None = None,
    ref_hline: float | None = None,
    ref_hline_label: str | None = None,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    strip_jitter: bool = True,
    y_axis_kind: str = "rho",
    y_floor: float | None = None,
    log_y_clip_upper: float | None = None,
    uniform_p_opt_hline_ns: tuple[int, ...] = (),
    uniform_qubo_p_opt_hline_ns: tuple[int, ...] = (),
    uniform_refs_in_ylim: bool = True,
) -> Any:
    """Boxplots vs QAOA depth *p* (dodged by series).

    ``y_axis_kind="rho"``: limits/ticks for approximation ratio; ``"generic"`` uses
    :func:`_decorate_y_axis_from_values` (e.g. :math:`P(\\mathrm{opt})`).

    ``uniform_p_opt_hline_ns``: draw horizontal reference :math:`1/(n-1)^{n-1}` per :math:`n` (TQUDO basis).
    ``uniform_qubo_p_opt_hline_ns``: draw horizontal reference :math:`1/2^{(n-1)^2}` per :math:`n` (QUBO).
    Uniform lines use the same face color as the **TQUDO qudits** and **QUBO** box series when those
    labels exist in *series*.

    If ``uniform_refs_in_ylim`` is false (log *y* only), reference masses are not mixed into *y* limits
    so the axis follows sample/box data (reference lines may fall outside the visible range).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]

    active = [(lab, dct) for lab, dct in series if dct]
    if not active:
        ax.set_xlabel(r"$p$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(
            y_label if y_label is not None else r"$\rho$",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        fig.tight_layout()
        return fig

    all_depths = sorted({d for _, dct in active for d in dct})
    n_s = len(active)
    rank_by_label: dict[str, int] = {lab: r for r, (lab, _) in enumerate(active)}
    r_tq = rank_by_label.get("TQUDO qudits")
    r_qb = rank_by_label.get("QUBO")
    col_tqudo_uni = colors[(r_tq if r_tq is not None else 0) % len(colors)]
    col_qubo_uni = colors[
        (r_qb if r_qb is not None else max(n_s - 1, 0)) % len(colors)
    ]
    if_narrow = max(n_s - 1, 1)
    if n_s <= 3:
        dodge = 0.09
    else:
        dodge = 0.43 / float(if_narrow)
    half = 0.5 * float(n_s - 1) if n_s > 1 else 0.0
    box_w_rel = 0.88
    jitter_w = min(dodge * box_w_rel * 0.88, 0.075)
    legend_handles: list[Any] = []
    all_rho: list[float] = []

    for rank, (label, dct) in enumerate(active):
        color = colors[rank % len(colors)]
        positions: list[float] = []
        col_data: list[list[float]] = []
        for d in all_depths:
            vals = dct.get(int(d), [])
            if not vals:
                continue
            if y_scale == "log" and y_axis_kind == "generic":
                vals = _clip_values_for_log_y(list(vals))
            positions.append(float(d) + (float(rank) - half) * dodge)
            col_data.append(vals)
            all_rho.extend(float(v) for v in vals if np.isfinite(v))
        if not col_data:
            continue
        bp = ax.boxplot(
            col_data,
            positions=positions,
            widths=dodge * box_w_rel,
            patch_artist=True,
            showfliers=not strip_jitter,
            boxprops=dict(linewidth=1.0, edgecolor=color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markersize=3,
                alpha=0.45,
                linestyle="none",
            ),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_zorder(2)
        if strip_jitter:
            for pos, vals in zip(positions, col_data, strict=True):
                seed = 9001 + rank * 131 + int(round(pos * 1000)) + len(vals)
                _scatter_rho_instances_jittered(
                    ax, pos, vals, color=color, jitter_span=jitter_w, rng=np.random.default_rng(seed)
                )
        legend_handles.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label)
        )

    if ref_hline is not None and math.isfinite(float(ref_hline)):
        if y_scale != "log" or float(ref_hline) > 0.0:
            rlab = ref_hline_label if ref_hline_label is not None else r"$\rho = 1$"
            ax.axhline(float(ref_hline), color="gray", linestyle="--", linewidth=1, zorder=1)
            legend_handles.append(
                Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label=rlab)
            )

    if uniform_refs_in_ylim:
        for un in uniform_p_opt_hline_ns:
            pu = _uniform_superposition_p_opt_htsp(int(un))
            if np.isfinite(pu) and pu > 0.0:
                all_rho.append(float(pu))
        for un in uniform_qubo_p_opt_hline_ns:
            pq = _uniform_superposition_p_opt_qubo(int(un))
            if np.isfinite(pq) and pq > 0.0:
                all_rho.append(float(pq))

    if all_depths:
        ax.set_xticks([float(d) for d in all_depths])
        ax.set_xticklabels([str(d) for d in all_depths])
    ax.set_xlabel(r"$p$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        y_label if y_label is not None else r"$\rho$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    if y_axis_kind == "rho":
        lo, hi = _ylim_rho_plot(all_rho, y_scale=y_scale)
        _decorate_approx_ratio_y_axis(
            ax, lo, hi, y_scale=y_scale, symlog_linthresh=symlog_linthresh
        )
        ax.set_ylim(lo, hi)
    else:
        _decorate_y_axis_from_values(
            ax,
            all_rho,
            y_scale=y_scale,
            symlog_linthresh=symlog_linthresh,
            log_y_clip_upper=log_y_clip_upper,
        )
        if y_floor is not None:
            y_lo = float(y_floor)
            if y_scale == "log" and y_lo <= 0.0:
                y_lo = float(_P_OPT_LOG_AXIS_FLOOR)
            _, cur_hi = ax.get_ylim()
            ax.set_ylim(y_lo, cur_hi)
    for un in uniform_p_opt_hline_ns:
        pu = _uniform_superposition_p_opt_htsp(int(un))
        if not (np.isfinite(pu) and pu > 0.0):
            continue
        ax.axhline(
            float(pu),
            color=col_tqudo_uni,
            linestyle="--",
            linewidth=1.25,
            alpha=0.9,
            zorder=1,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=col_tqudo_uni,
                linestyle="--",
                linewidth=1.25,
                label=rf"Uniform TQUDO $n={un}$ ($1/{un - 1}^{{{un - 1}}}$)",
            )
        )
    for un in uniform_qubo_p_opt_hline_ns:
        pq = _uniform_superposition_p_opt_qubo(int(un))
        if not (np.isfinite(pq) and pq > 0.0):
            continue
        ax.axhline(
            float(pq),
            color=col_qubo_uni,
            linestyle=":",
            linewidth=1.4,
            alpha=0.95,
            zorder=1,
        )
        exp = (int(un) - 1) ** 2
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=col_qubo_uni,
                linestyle=":",
                linewidth=1.4,
                label=rf"Uniform QUBO $n={un}$ ($1/2^{{{exp}}}$)",
            )
        )
    if (
        uniform_refs_in_ylim
        and y_axis_kind == "generic"
        and y_scale == "log"
        and (uniform_p_opt_hline_ns or uniform_qubo_p_opt_hline_ns)
    ):
        ref_lo: list[float] = []
        for un in uniform_p_opt_hline_ns:
            pu = _uniform_superposition_p_opt_htsp(int(un))
            if np.isfinite(pu) and pu > 0.0:
                ref_lo.append(float(pu))
        for un in uniform_qubo_p_opt_hline_ns:
            pq = _uniform_superposition_p_opt_qubo(int(un))
            if np.isfinite(pq) and pq > 0.0:
                ref_lo.append(float(pq))
        if ref_lo:
            y_a, y_b = ax.get_ylim()
            ax.set_ylim(min(y_a, min(ref_lo) * 0.35), y_b)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE_COMPACT if len(legend_handles) > 5 else LEGEND_FONTSIZE,
        )
    fig.tight_layout()
    return fig


def _plot_approx_ratio_boxplots_vs_ncities(
    series: list[tuple[str, list[float], list[list[float]]]],
    *,
    n_tick_vals: list[int],
    figsize: tuple[float, float] = (8.0, 5.0),
    y_label: str | None = None,
    ref_hline: float | None = None,
    ref_hline_label: str | None = None,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    strip_jitter: bool = True,
) -> Any:
    """``series``: (label, x positions, one list of ρ per box, aligned with *x*)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    legend_handles: list[Any] = []
    box_w = 0.1
    jitter_w = min(box_w * 0.78, 0.07)
    all_rho: list[float] = []

    for i, (label, xs, datas) in enumerate(series):
        if not xs or not datas or len(xs) != len(datas):
            continue
        color = colors[i % len(colors)]
        for vals in datas:
            all_rho.extend(float(v) for v in vals if np.isfinite(v))
        bp = ax.boxplot(
            datas,
            positions=xs,
            widths=box_w,
            patch_artist=True,
            showfliers=not strip_jitter,
            boxprops=dict(linewidth=1.0, edgecolor=color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markersize=3,
                alpha=0.45,
                linestyle="none",
            ),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_zorder(2)
        if strip_jitter:
            for x_pos, vals in zip(xs, datas, strict=True):
                seed = 4242 + i * 97 + int(round(x_pos * 1000)) + len(vals)
                _scatter_rho_instances_jittered(
                    ax,
                    float(x_pos),
                    vals,
                    color=color,
                    jitter_span=jitter_w,
                    rng=np.random.default_rng(seed),
                )
        legend_handles.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label)
        )

    if ref_hline is not None and math.isfinite(float(ref_hline)):
        if y_scale != "log" or float(ref_hline) > 0.0:
            rlab = ref_hline_label if ref_hline_label is not None else r"$\rho = 1$"
            ax.axhline(float(ref_hline), color="gray", linestyle="--", linewidth=1, zorder=1)
            legend_handles.append(
                Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label=rlab)
            )

    ax.set_xticks([float(n) for n in n_tick_vals])
    ax.set_xticklabels([str(n) for n in n_tick_vals])
    ax.set_xlabel(r"$n$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        y_label if y_label is not None else r"$\rho$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    lo, hi = _ylim_rho_plot(all_rho, y_scale=y_scale)
    _decorate_approx_ratio_y_axis(
        ax, lo, hi, y_scale=y_scale, symlog_linthresh=symlog_linthresh
    )
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE_COMPACT if len(legend_handles) > 5 else LEGEND_FONTSIZE,
        )
    fig.tight_layout()
    return fig


def _plot_dodged_boxplot_series_vs_ncities(
    series: list[tuple[str, list[float], list[list[float]]]],
    *,
    n_tick_vals: list[int],
    xlabel: str = r"$n$",
    y_label: str,
    figsize: tuple[float, float] = (8.5, 5.0),
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    asinh_linear_width: float | None = None,
    strip_jitter: bool = True,
    y_floor: float | None = None,
    log_y_clip_upper: float | None = None,
    manual_y_limits: bool = True,
    uniform_p_opt_vline_ns: list[int] | None = None,
) -> Any:
    """Dodged box/strip series vs city count (same layout as ρ vs *n*; generic *y* axis).

    If ``uniform_p_opt_vline_ns`` is set, draw a dashed polyline through each
    reference point :math:`(n, 1/(n-1)^{n-1})` (markers at every :math:`n` so all stay
    visible on log *y*).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    legend_handles: list[Any] = []
    box_w = 0.1
    jitter_w = min(box_w * 0.78, 0.07)
    all_y: list[float] = []

    for i, (label, xs, datas) in enumerate(series):
        if not xs or not datas or len(xs) != len(datas):
            continue
        color = colors[i % len(colors)]
        draw_lists = (
            [_clip_values_for_log_y(list(v)) for v in datas] if y_scale == "log" else datas
        )
        for vals in draw_lists:
            all_y.extend(float(v) for v in vals if np.isfinite(v))
        bp = ax.boxplot(
            draw_lists,
            positions=xs,
            widths=box_w,
            patch_artist=True,
            showfliers=not strip_jitter,
            boxprops=dict(linewidth=1.0, edgecolor=color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markersize=3,
                alpha=0.45,
                linestyle="none",
            ),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_zorder(2)
        if strip_jitter:
            for x_pos, vals in zip(xs, draw_lists, strict=True):
                seed = 8080 + i * 97 + int(round(x_pos * 1000)) + len(vals)
                _scatter_rho_instances_jittered(
                    ax,
                    float(x_pos),
                    vals,
                    color=color,
                    jitter_span=jitter_w,
                    rng=np.random.default_rng(seed),
                )
        legend_handles.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label)
        )

    if uniform_p_opt_vline_ns:
        for n in uniform_p_opt_vline_ns:
            pu = _uniform_superposition_p_opt_htsp(int(n))
            if np.isfinite(pu) and pu > 0.0:
                all_y.append(float(pu))

    ax.set_xticks([float(n) for n in n_tick_vals])
    ax.set_xticklabels([str(n) for n in n_tick_vals])
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    _decorate_y_axis_from_values(
        ax,
        all_y,
        y_scale=y_scale,
        symlog_linthresh=symlog_linthresh,
        asinh_linear_width=asinh_linear_width,
        log_y_clip_upper=log_y_clip_upper,
        manual_y_limits=manual_y_limits,
    )
    if y_floor is not None:
        y_lo = float(y_floor)
        if y_scale == "log" and y_lo <= 0.0:
            y_lo = float(_P_OPT_LOG_AXIS_FLOOR)
        _, cur_hi = ax.get_ylim()
        ax.set_ylim(y_lo, cur_hi)
    if uniform_p_opt_vline_ns:
        ns_sorted = sorted({int(x) for x in uniform_p_opt_vline_ns})
        xs_u: list[float] = []
        ys_u: list[float] = []
        for n in ns_sorted:
            pu = _uniform_superposition_p_opt_htsp(n)
            if np.isfinite(pu) and pu > 0.0:
                xs_u.append(float(n))
                ys_u.append(float(pu))
        if xs_u:
            ax.plot(
                xs_u,
                ys_u,
                linestyle="--",
                color="dimgray",
                linewidth=1.45,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor="dimgray",
                markeredgewidth=1.2,
                alpha=0.95,
                zorder=4,
                clip_on=False,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="dimgray",
                    linewidth=1.45,
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor="dimgray",
                    markeredgewidth=1.2,
                    label=r"Uniform $P=1/(n-1)^{n-1}$",
                )
            )
            if y_scale == "log" and ys_u:
                y_lo_c, y_hi_c = ax.get_ylim()
                y_min_ref = float(min(ys_u))
                y_max_ref = float(max(ys_u))
                ax.set_ylim(
                    min(y_lo_c, y_min_ref / 1.35),
                    max(y_hi_c, y_max_ref * 1.08, 1.0),
                )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE_COMPACT if len(legend_handles) > 5 else LEGEND_FONTSIZE,
        )
    fig.tight_layout()
    return fig


def _plot_paired_four_series_boxplots_vs_p(
    *,
    x_labels: list[str],
    series: list[tuple[str, list[list[float]]]],
    y_label: str,
    x_axis_label: str,
    figsize: tuple[float, float] = (9.5, 5.2),
    strip_jitter: bool = True,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
) -> Any:
    """Up to four dodged box/strip series per *p* (e.g. qubits/qudits × two *n*)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    n_g = len(x_labels)
    n_s = len(series)
    if_n = max(n_s - 1, 1)
    if n_s <= 3:
        dodge = 0.09
    else:
        dodge = 0.43 / float(if_n)
    half = 0.5 * float(n_s - 1) if n_s > 1 else 0.0
    box_w = dodge * 0.88
    jitter_w = min(box_w * 0.82, 0.055)
    all_y: list[float] = []
    legend_handles: list[Any] = []

    for rank, (leg_label, values_rows) in enumerate(series):
        color = colors[rank % len(colors)]
        legend_handles.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.45, label=leg_label)
        )
        for i in range(n_g):
            vals = values_rows[i] if i < len(values_rows) else []
            pos = float(i) + (float(rank) - half) * dodge
            for v in vals:
                if np.isfinite(v):
                    all_y.append(float(v))
            if not vals:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_w,
                patch_artist=True,
                showfliers=not strip_jitter,
                boxprops=dict(linewidth=1.0, edgecolor=color),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color=color, linewidth=1.0),
                capprops=dict(color=color, linewidth=1.0),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=color,
                    markersize=3,
                    alpha=0.45,
                    linestyle="none",
                ),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
                patch.set_zorder(2)
            if strip_jitter:
                _scatter_rho_instances_jittered(
                    ax,
                    pos,
                    vals,
                    color=color,
                    jitter_span=jitter_w,
                    rng=np.random.default_rng(8800 + rank * 41 + i * 19 + len(vals)),
                )

    ax.set_xticks([float(i) for i in range(n_g)])
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    _decorate_y_axis_from_values(
        ax, all_y, y_scale=y_scale, symlog_linthresh=symlog_linthresh
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(handles=legend_handles, fontsize=LEGEND_FONTSIZE_COMPACT)
    fig.tight_layout()
    return fig


def _p_opt_final_from_row(
    row: Any,
    output_root: Path,
    formulation: str,
    bf_cache: dict[tuple[int, int], list[int] | None],
) -> float | None:
    if not coerce_bool_scalar(row.get("parse_ok")) or not coerce_bool_scalar(row.get("solve_ok")):
        return None
    if not coerce_bool_scalar(row.get("has_final_samples")):
        return None
    rel = row.get("path")
    if rel is None:
        return None
    s = str(rel).strip()
    if not s or s.lower() == "nan":
        return None
    n_cities = int(row["n_cities"])
    instance_key = int(row["instance_key"])
    seq = load_bruteforce_optimal_sequence(
        output_root, n_cities, instance_key, cache=bf_cache
    )
    if seq is None:
        return None
    key = histogram_key_for_formulation(seq, formulation, n_cities)
    _, fin = read_sample_histograms_from_solution_json(output_root / s)
    return histogram_mass(fin, key)


def _delta_p_opt_from_row(
    row: Any,
    output_root: Path,
    formulation: str,
    bf_cache: dict[tuple[int, int], list[int] | None],
) -> float | None:
    if not coerce_bool_scalar(row.get("parse_ok")) or not coerce_bool_scalar(row.get("solve_ok")):
        return None
    if not coerce_bool_scalar(row.get("has_final_samples")) or not coerce_bool_scalar(
        row.get("has_initial_samples")
    ):
        return None
    rel = row.get("path")
    if rel is None:
        return None
    s = str(rel).strip()
    if not s or s.lower() == "nan":
        return None
    n_cities = int(row["n_cities"])
    instance_key = int(row["instance_key"])
    seq = load_bruteforce_optimal_sequence(
        output_root, n_cities, instance_key, cache=bf_cache
    )
    if seq is None:
        return None
    key = histogram_key_for_formulation(seq, formulation, n_cities)
    init, fin = read_sample_histograms_from_solution_json(output_root / s)
    p0 = histogram_mass(init, key)
    p1 = histogram_mass(fin, key)
    if p0 is None or p1 is None:
        return None
    return float(p1 - p0)


def _p_opt_lists_by_depth_unpaired(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
    output_root: Path,
    bf_cache: dict[tuple[int, int], list[int] | None],
) -> dict[int, list[float]]:
    """Per QAOA depth: list of :math:`P(\\mathrm{opt})` from final samples (finite only)."""
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == n_cities)
        & paired["has_final_samples"]
        & paired["qaoa_depth"].notna()
    )
    sub = paired.loc[m].copy()
    if sub.empty:
        return {}
    sub = _dedupe_solution_rows(
        sub,
        ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
    )
    sub["qaoa_depth"] = sub["qaoa_depth"].astype(int)
    vals_by_d: dict[int, list[float]] = {}
    for _, row in sub.iterrows():
        p = _p_opt_final_from_row(row, output_root, formulation, bf_cache)
        if p is None:
            continue
        pf = float(p)
        if not np.isfinite(pf):
            continue
        d = int(row["qaoa_depth"])
        vals_by_d.setdefault(d, []).append(pf)
    return vals_by_d


def _collect_numeric_by_ncities_depth(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    depth_values: tuple[int, ...],
    output_root: Path,
    value_fn: Any,
    bf_cache: dict[tuple[int, int], list[int] | None],
) -> list[tuple[str, list[float], list[float], list[float]]]:
    """Errorbar series per depth: x = n_cities (with dodge), y = mean(value), yerr = std.

    ``value_fn(row, output_root, formulation, bf_cache)`` returns a float or ``None``.
    """
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & paired["qaoa_depth"].notna()
    )
    sub0 = paired.loc[m].copy()
    if sub0.empty:
        return []
    sub0["qaoa_depth"] = sub0["qaoa_depth"].astype(int)
    sub0 = sub0[sub0["qaoa_depth"].isin([int(x) for x in depth_values])]
    if sub0.empty:
        return []
    n_vals = sorted({int(x) for x in sub0["n_cities"].unique() if pd_notna_n(x)})
    if not n_vals:
        return []

    depths_sorted = [int(d) for d in depth_values if int(d) in set(sub0["qaoa_depth"].unique())]
    if not depths_sorted:
        return []
    dodge_step = 0.14
    half = 0.5 * float(len(depths_sorted) - 1) if len(depths_sorted) > 1 else 0.0
    out: list[tuple[str, list[float], list[float], list[float]]] = []
    for rank, depth in enumerate(depths_sorted):
        x_off = (float(rank) - half) * dodge_step if len(depths_sorted) > 1 else 0.0
        xs: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        for n in n_vals:
            sel = sub0[
                (sub0["n_cities"] == n) & (sub0["qaoa_depth"] == depth)
            ].copy()
            sel = _dedupe_solution_rows(
                sel,
                ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
            )
            collected: list[float] = []
            for _, row in sel.iterrows():
                v = value_fn(row, output_root, formulation, bf_cache)
                if v is not None and v == v:
                    collected.append(float(v))
            if not collected:
                continue
            a = np.asarray(collected, dtype=np.float64)
            xs.append(float(n) + x_off)
            means.append(float(a.mean()))
            stds.append(float(a.std(ddof=1)) if a.size > 1 else 0.0)
        if xs:
            out.append((f"p = {depth}", xs, means, stds))
    return out


def _collect_numeric_box_series_vs_ncities(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    depth_values: tuple[int, ...],
    output_root: Path,
    value_fn: Any,
    bf_cache: dict[tuple[int, int], list[int] | None],
) -> list[tuple[str, list[float], list[list[float]]]]:
    """Boxplot series per depth: x = n_cities (dodged), raw per-instance values.

    ``value_fn(row, output_root, formulation, bf_cache)`` -> float or ``None`` (same as
    :func:`_collect_numeric_by_ncities_depth`).
    """
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & paired["qaoa_depth"].notna()
    )
    sub0 = paired.loc[m].copy()
    if sub0.empty:
        return []
    sub0["qaoa_depth"] = sub0["qaoa_depth"].astype(int)
    sub0 = sub0[sub0["qaoa_depth"].isin([int(x) for x in depth_values])]
    if sub0.empty:
        return []
    n_vals = sorted({int(x) for x in sub0["n_cities"].unique() if pd_notna_n(x)})
    if not n_vals:
        return []

    depths_sorted = [int(d) for d in depth_values if int(d) in set(sub0["qaoa_depth"].unique())]
    if not depths_sorted:
        return []
    dodge_step = 0.14
    half = 0.5 * float(len(depths_sorted) - 1) if len(depths_sorted) > 1 else 0.0
    out: list[tuple[str, list[float], list[list[float]]]] = []
    for rank, depth in enumerate(depths_sorted):
        x_off = (float(rank) - half) * dodge_step if len(depths_sorted) > 1 else 0.0
        xs: list[float] = []
        datas: list[list[float]] = []
        for n in n_vals:
            sel = sub0[
                (sub0["n_cities"] == n) & (sub0["qaoa_depth"] == depth)
            ].copy()
            sel = _dedupe_solution_rows(
                sel,
                ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
            )
            collected: list[float] = []
            for _, row in sel.iterrows():
                v = value_fn(row, output_root, formulation, bf_cache)
                if v is None or v != v:
                    continue
                vf = float(v)
                if np.isfinite(vf):
                    collected.append(vf)
            if not collected:
                continue
            xs.append(float(n) + x_off)
            datas.append(collected)
        if xs:
            out.append((f"p = {depth}", xs, datas))
    return out


def pd_notna_n(x: Any) -> bool:
    """True if *x* is a usable ``n_cities`` scalar (finite int)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return xf == xf and abs(xf) < 1e100


def _collect_energy_improvement_box_series_vs_ncities(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    depth_values: tuple[int, ...],
) -> list[tuple[str, list[float], list[list[float]]]]:
    """Raw ``energy_improvement_rel`` lists per (``n_cities``, depth), for boxplots."""
    if "energy_improvement_rel" not in paired.columns:
        return []
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & paired["qaoa_depth"].notna()
    )
    sub0 = paired.loc[m].copy()
    if sub0.empty:
        return []
    sub0["qaoa_depth"] = sub0["qaoa_depth"].astype(int)
    sub0 = sub0[sub0["qaoa_depth"].isin([int(x) for x in depth_values])]
    if sub0.empty:
        return []
    n_vals = sorted({int(x) for x in sub0["n_cities"].unique() if pd_notna_n(x)})
    depths_sorted = [int(d) for d in depth_values if int(d) in set(sub0["qaoa_depth"].unique())]
    if not n_vals or not depths_sorted:
        return []
    dodge_step = 0.14
    half = 0.5 * float(len(depths_sorted) - 1) if len(depths_sorted) > 1 else 0.0
    out: list[tuple[str, list[float], list[list[float]]]] = []
    for rank, depth in enumerate(depths_sorted):
        x_off = (float(rank) - half) * dodge_step if len(depths_sorted) > 1 else 0.0
        xs: list[float] = []
        datas: list[list[float]] = []
        for n in n_vals:
            sel = sub0[
                (sub0["n_cities"] == n) & (sub0["qaoa_depth"] == depth)
            ].copy()
            sel = _dedupe_solution_rows(
                sel,
                ["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
            )
            col = sel["energy_improvement_rel"].dropna().to_numpy(dtype=np.float64)
            col = col[np.isfinite(col)]
            vals = [float(v) for v in col]
            if not vals:
                continue
            xs.append(float(n) + x_off)
            datas.append(vals)
        if xs:
            out.append((f"p = {depth}", xs, datas))
    return out


def _plot_grouped_bars_mean_std(
    *,
    x_labels: list[str],
    means_left: list[float],
    stds_left: list[float],
    means_right: list[float],
    stds_right: list[float],
    label_left: str,
    label_right: str,
    y_label: str,
    x_axis_label: str,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
) -> Any:
    """Grouped bars with error bars (same layout as opt-steps plot).

    ``y_scale``: ``"linear"``, ``"log"`` (bars clipped upward to a small floor), or ``"symlog"``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(x_labels), dtype=np.float64)
    w = 0.36
    ml = np.array([float(v) if v == v else 0.0 for v in means_left], dtype=np.float64)
    mr = np.array([float(v) if v == v else 0.0 for v in means_right], dtype=np.float64)
    el = np.asarray(stds_left, dtype=np.float64)
    er = np.asarray(stds_right, dtype=np.float64)
    if y_scale == "log":
        ml = np.maximum(ml, _LOG_Y_FLOOR)
        mr = np.maximum(mr, _LOG_Y_FLOOR)
        el = np.minimum(el, np.maximum(ml - _LOG_Y_FLOOR, 0.0))
        er = np.minimum(er, np.maximum(mr - _LOG_Y_FLOOR, 0.0))
    ax.bar(x - w / 2, ml, w, yerr=el, capsize=3, label=label_left, color="#1f77b4")
    ax.bar(x + w / 2, mr, w, yerr=er, capsize=3, label=label_right, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    if y_scale == "log":
        ax.set_yscale("log")
    elif y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh, base=10)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    return fig


def _paired_metric_lists_by_depth(
    merged: Any,
    *,
    depths: tuple[int, ...],
    col_left: str,
    col_right: str,
) -> tuple[list[list[float]], list[list[float]]]:
    """Per depth: finite lists of left/right column values (paired rows)."""
    lists_l: list[list[float]] = []
    lists_r: list[list[float]] = []
    if merged.empty or col_left not in merged.columns or col_right not in merged.columns:
        return ([[] for _ in depths], [[] for _ in depths])
    for d in depths:
        sub = merged[_mask_qaoa_depth_eq(merged["qaoa_depth"], int(d))]
        vl = sub[col_left].dropna().to_numpy(dtype=np.float64)
        vr = sub[col_right].dropna().to_numpy(dtype=np.float64)
        vl = vl[np.isfinite(vl)]
        vr = vr[np.isfinite(vr)]
        lists_l.append([float(x) for x in vl])
        lists_r.append([float(x) for x in vr])
    return lists_l, lists_r


def _paired_delta_p_opt_lists_by_depth(
    merged: Any,
    *,
    depths: tuple[int, ...],
    output_root: Path,
    bf_cache: dict[tuple[int, int], list[int] | None],
) -> tuple[list[list[float]], list[list[float]]]:
    """Per depth: lists of :math:`\\Delta P(\\mathrm{opt})` (paired rows, final − initial)."""
    lists_l: list[list[float]] = []
    lists_r: list[list[float]] = []
    if merged.empty or "path_left" not in merged.columns:
        return ([[] for _ in depths], [[] for _ in depths])
    for d in depths:
        sub = merged[_mask_qaoa_depth_eq(merged["qaoa_depth"], int(d))]
        dl: list[float] = []
        dr: list[float] = []
        for _, row in sub.iterrows():
            pl = row.get("path_left")
            pr = row.get("path_right")
            if pl is None or pr is None:
                continue
            n_cities = int(row["n_cities"])
            ik = int(row["instance_key"])
            seq = load_bruteforce_optimal_sequence(
                output_root, n_cities, ik, cache=bf_cache
            )
            if seq is None:
                continue
            key_l = histogram_key_for_formulation(seq, "tqudo_virtual", n_cities)
            key_r = histogram_key_for_formulation(seq, "tqudo", n_cities)
            init_l, fin_l = read_sample_histograms_from_solution_json(
                output_root / str(pl).strip()
            )
            init_r, fin_r = read_sample_histograms_from_solution_json(
                output_root / str(pr).strip()
            )
            p0l = histogram_mass(init_l, key_l)
            p1l = histogram_mass(fin_l, key_l)
            p0r = histogram_mass(init_r, key_r)
            p1r = histogram_mass(fin_r, key_r)
            if p0l is None or p1l is None or p0r is None or p1r is None:
                continue
            dl.append(float(p1l - p0l))
            dr.append(float(p1r - p0r))
        lists_l.append(dl)
        lists_r.append(dr)
    return lists_l, lists_r


def run_benchmark_plots(paired: Any, output_root: Path, images_dir: Path) -> None:
    """Write CUDA-Q / cross-backend dashboard and mean approximation-ratio figures."""
    import matplotlib.pyplot as plt

    images_dir.mkdir(parents=True, exist_ok=True)
    d_dash = images_dir / "dashboards"
    d_rho = images_dir / "approx_ratio"
    d_steps = images_dir / "steps"
    d_impr = images_dir / "improvement"
    d_popt = images_dir / "p_opt"
    for d in (d_dash, d_rho, d_steps, d_impr, d_popt):
        d.mkdir(parents=True, exist_ok=True)
    root = output_root.resolve()
    bf_cache: dict[tuple[int, int], list[int] | None] = {}

    # --- CUDA-Q: QUBO vs TQUDO qubits by QAOA depth (n_cities = 5) ---
    depths = (1, 2, 3)
    cq_merged = _merge_paired(
        paired,
        left=("cudaq", "qubo"),
        right=("cudaq", "tqudo_virtual"),
        dedupe_keys=["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
        merge_on=["n_cities", "instance_key", "qaoa_depth"],
        n_cities_filter=5,
    )
    x_labels = [str(d) for d in depths]
    stats_list = []
    for d in depths:
        if cq_merged.empty:
            stats_list.append(_stats_from_rows(cq_merged.iloc[0:0]))
        else:
            sub = cq_merged[cq_merged["qaoa_depth"].astype(int) == int(d)]
            stats_list.append(_stats_from_rows(sub))
    fig = _plot_comparison_dashboard(
        x_labels=x_labels,
        stats_list=stats_list,
        label_left="QUBO",
        label_right="TQUDO qubits",
        x_axis_label=r"$p$",
    )
    fig.savefig(
        d_dash / "cudaq_qubo_vs_tvirt_n5.png",
        dpi=150,
    )
    plt.close(fig)

    # --- TQUDO qubits (CUDA-Q) vs TQUDO qudits (Cirq) by QAOA depth ---
    xc_merged = _merge_paired(
        paired,
        left=("cudaq", "tqudo_virtual"),
        right=("cirq", "tqudo"),
        dedupe_keys=["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
        merge_on=["n_cities", "instance_key", "qaoa_depth"],
        n_cities_filter=5,
    )
    stats_list_x = []
    for d in depths:
        if xc_merged.empty:
            stats_list_x.append(_stats_from_rows(xc_merged.iloc[0:0]))
        else:
            sub = xc_merged[xc_merged["qaoa_depth"].astype(int) == int(d)]
            stats_list_x.append(_stats_from_rows(sub))
    fig = _plot_comparison_dashboard(
        x_labels=x_labels,
        stats_list=stats_list_x,
        label_left="TQUDO qubits",
        label_right="TQUDO qudits",
        x_axis_label=r"$p$",
    )
    fig.savefig(
        d_dash / "cudaq_tvirt_vs_cirq_n5.png",
        dpi=150,
    )
    plt.close(fig)

    # --- TQUDO qubits (CUDA-Q) vs TQUDO qudits (Cirq) by QAOA depth (n = 9) ---
    xc_merged_n9 = _merge_paired(
        paired,
        left=("cudaq", "tqudo_virtual"),
        right=("cirq", "tqudo"),
        dedupe_keys=["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
        merge_on=["n_cities", "instance_key", "qaoa_depth"],
        n_cities_filter=9,
    )
    stats_list_x9: list[dict[str, float | int]] = []
    for d in depths:
        if xc_merged_n9.empty:
            stats_list_x9.append(_stats_from_rows(xc_merged_n9.iloc[0:0]))
        else:
            sub9 = xc_merged_n9[xc_merged_n9["qaoa_depth"].astype(int) == int(d)]
            stats_list_x9.append(_stats_from_rows(sub9))
    fig = _plot_comparison_dashboard(
        x_labels=x_labels,
        stats_list=stats_list_x9,
        label_left="TQUDO qubits",
        label_right="TQUDO qudits",
        x_axis_label=r"$p$",
    )
    fig.savefig(
        d_dash / "cudaq_tvirt_vs_cirq_n9.png",
        dpi=150,
    )
    plt.close(fig)

    # --- Approximation ratio vs p: n=5 (QUBO + TQUDO) and n=9 (TQUDO virt + native only) ---
    rho_q = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cudaq", formulation="qubo", n_cities=5
    )
    rho_tv5 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cudaq", formulation="tqudo_virtual", n_cities=5
    )
    rho_ci5 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cirq", formulation="tqudo", n_cities=5
    )
    rho_tv9 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cudaq", formulation="tqudo_virtual", n_cities=9
    )
    rho_ci9 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cirq", formulation="tqudo", n_cities=9
    )
    fig = _plot_approx_ratio_boxplots_vs_p(
        [
            (r"QUBO ($n=5$)", rho_q),
            (r"TQUDO qubits ($n=5$)", rho_tv5),
            (r"TQUDO qudits ($n=5$)", rho_ci5),
            (r"TQUDO qubits ($n=9$)", rho_tv9),
            (r"TQUDO qudits ($n=9$)", rho_ci9),
        ],
        figsize=(7.8, 7.8),
    )
    fig.savefig(
        d_rho / "n5_qubo_tvirt_cirq_vs_p.png",
        dpi=150,
    )
    plt.close(fig)

    n_multi = [5, 6, 7, 8, 9]
    vs_n_box = _approx_ratio_box_series_vs_ncities_by_depth(paired, n_values=n_multi)
    fig = _plot_approx_ratio_boxplots_vs_ncities(
        vs_n_box,
        n_tick_vals=n_multi,
        figsize=(8.5, 5),
    )
    fig.savefig(
        d_rho / "rho_vs_n_by_p.png",
        dpi=150,
    )
    plt.close(fig)

    # --- Steps to first trace minimum: optimal on that solver only (paired instances) ---
    tvirt_qubo_merged = _merge_paired(
        paired,
        left=("cudaq", "tqudo_virtual"),
        right=("cudaq", "qubo"),
        dedupe_keys=["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
        merge_on=["n_cities", "instance_key", "qaoa_depth"],
        n_cities_filter=5,
    )
    ll_cq, lr_cq = _collect_side_opt_step_lists_by_depth(
        tvirt_qubo_merged, depths=depths, output_root=root
    )
    fig_cq = _plot_approx_ratio_boxplots_vs_p(
        [
            ("QUBO", _step_lists_to_depth_dict(depths, lr_cq)),
            ("TQUDO qubits", _step_lists_to_depth_dict(depths, ll_cq)),
        ],
        y_label="steps",
        y_axis_kind="generic",
        y_scale="linear",
        figsize=(6.9, 6.9),
    )
    fig_cq.savefig(
        d_steps / "cudaq_tvirt_vs_qubo_n5_vs_p.png",
        dpi=150,
    )
    plt.close(fig_cq)

    ll5, lr5 = _collect_side_opt_step_lists_by_depth(
        xc_merged, depths=depths, output_root=root
    )
    ll9, lr9 = _collect_side_opt_step_lists_by_depth(
        xc_merged_n9, depths=depths, output_root=root
    )
    fig_xc = _plot_approx_ratio_boxplots_vs_p(
        [
            (r"TQUDO qubits ($n=5$)", _step_lists_to_depth_dict(depths, ll5)),
            (r"TQUDO qudits ($n=5$)", _step_lists_to_depth_dict(depths, lr5)),
            (r"TQUDO qubits ($n=9$)", _step_lists_to_depth_dict(depths, ll9)),
            (r"TQUDO qudits ($n=9$)", _step_lists_to_depth_dict(depths, lr9)),
        ],
        y_label="steps",
        y_axis_kind="generic",
        y_scale="linear",
        figsize=(6.9, 6.9),
    )
    fig_xc.savefig(
        d_steps / "cudaq_tvirt_vs_cirq_n5_n9_vs_p.png",
        dpi=150,
    )
    plt.close(fig_xc)

    opt_steps_nc_box = _collect_cirq_tqudo_opt_steps_box_series_vs_ncities(
        paired,
        n_values=[5, 6, 7, 8, 9],
        depth_values=depths,
        output_root=root,
    )
    if opt_steps_nc_box:
        fig_ci = _plot_dodged_boxplot_series_vs_ncities(
            opt_steps_nc_box,
            n_tick_vals=[5, 6, 7, 8, 9],
            y_label="steps",
            figsize=(8.5, 5.0),
        )
        fig_ci.savefig(
            d_steps / "cirq_tqudo_firstmin_steps_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig_ci)

    # --- P(ground) vs n (TQUDO qudits / Cirq, p = 1,2,3) ---
    m_ct_n = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == "cirq")
        & (paired["formulation"] == "tqudo")
        & paired["qaoa_depth"].notna()
    )
    n_tick_cirq_tqudo = sorted(
        {int(x) for x in paired.loc[m_ct_n, "n_cities"].unique() if pd_notna_n(x)}
    )
    series_popt_n_box = _collect_numeric_box_series_vs_ncities(
        paired,
        solver="cirq",
        formulation="tqudo",
        depth_values=depths,
        output_root=root,
        value_fn=_p_opt_final_from_row,
        bf_cache=bf_cache,
    )
    if series_popt_n_box and n_tick_cirq_tqudo:
        fig = _plot_dodged_boxplot_series_vs_ncities(
            series_popt_n_box,
            n_tick_vals=n_tick_cirq_tqudo,
            y_label=r"$P(\mathrm{opt})$",
            figsize=(8.5, 5),
            y_scale="log",
            log_y_clip_upper=1.0,
            uniform_p_opt_vline_ns=list(n_tick_cirq_tqudo),
        )
        fig.savefig(
            d_popt / "cirq_tqudo_popt_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig)

    # --- P(ground) vs p, n = 5: QUBO then TQUDO (CUDA-Q qubits vs Cirq qudits) ---
    popt_ci5 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cirq",
        formulation="tqudo",
        n_cities=5,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_cq5 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cudaq",
        formulation="tqudo_virtual",
        n_cities=5,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_q5 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cudaq",
        formulation="qubo",
        n_cities=5,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_n5_series = [
        ("QUBO", popt_q5),
        ("TQUDO qubits", popt_cq5),
        ("TQUDO qudits", popt_ci5),
    ]
    fig_popt_full = _plot_approx_ratio_boxplots_vs_p(
        popt_n5_series,
        y_label=r"$P(\mathrm{opt})$",
        y_axis_kind="generic",
        y_scale="log",
        log_y_clip_upper=1.0,
        figsize=(6.9, 6.9),
        uniform_p_opt_hline_ns=(5, 9),
        uniform_qubo_p_opt_hline_ns=(5,),
        uniform_refs_in_ylim=True,
    )
    fig_popt_full.savefig(
        d_popt / "n5_cirq_vs_cq_tvirt_popt_vs_p.png",
        dpi=150,
    )
    plt.close(fig_popt_full)
    fig_popt_ydata = _plot_approx_ratio_boxplots_vs_p(
        popt_n5_series,
        y_label=r"$P(\mathrm{opt})$",
        y_axis_kind="generic",
        y_scale="log",
        log_y_clip_upper=1.0,
        figsize=(6.9, 6.9),
        uniform_p_opt_hline_ns=(5, 9),
        uniform_qubo_p_opt_hline_ns=(5,),
        uniform_refs_in_ylim=False,
    )
    fig_popt_ydata.savefig(
        d_popt / "n5_cirq_vs_cq_tvirt_popt_vs_p_ydata.png",
        dpi=150,
    )
    plt.close(fig_popt_ydata)

    # --- Relative energy improvement vs n (TQUDO qudits / Cirq), boxplots ---
    series_eimp_n_box = _collect_energy_improvement_box_series_vs_ncities(
        paired,
        solver="cirq",
        formulation="tqudo",
        depth_values=depths,
    )
    if series_eimp_n_box and n_tick_cirq_tqudo:
        fig = _plot_dodged_boxplot_series_vs_ncities(
            series_eimp_n_box,
            n_tick_vals=n_tick_cirq_tqudo,
            y_label=r"$(E_0 - E^\star) / |E_0|$",
            figsize=(8.5, 5),
        )
        fig.savefig(
            d_impr / "cirq_tqudo_rel_energy_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig)

    # --- Δ P(ground) vs n (TQUDO qudits / Cirq): final − initial histogram ---
    series_dp_n_box = _collect_numeric_box_series_vs_ncities(
        paired,
        solver="cirq",
        formulation="tqudo",
        depth_values=depths,
        output_root=root,
        value_fn=_delta_p_opt_from_row,
        bf_cache=bf_cache,
    )
    if series_dp_n_box and n_tick_cirq_tqudo:
        fig = _plot_dodged_boxplot_series_vs_ncities(
            series_dp_n_box,
            n_tick_vals=n_tick_cirq_tqudo,
            y_label=r"$\Delta P(\mathrm{opt})$",
            figsize=(8.5, 5),
            y_scale="asinh",
            symlog_linthresh=2e-3,
            manual_y_limits=False,
        )
        fig.savefig(
            d_popt / "cirq_tqudo_delta_popt_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig)

    # --- Paired n = 5 & 9: relative energy improvement vs *p* (four box series, one figure) ---
    eimp_l5, eimp_r5 = _paired_metric_lists_by_depth(
        xc_merged,
        depths=depths,
        col_left="energy_improvement_rel_left",
        col_right="energy_improvement_rel_right",
    )
    eimp_l9, eimp_r9 = _paired_metric_lists_by_depth(
        xc_merged_n9,
        depths=depths,
        col_left="energy_improvement_rel_left",
        col_right="energy_improvement_rel_right",
    )
    fig = _plot_paired_four_series_boxplots_vs_p(
        x_labels=x_labels,
        series=[
            (r"TQUDO qubits ($n=5$)", eimp_l5),
            (r"TQUDO qudits ($n=5$)", eimp_r5),
            (r"TQUDO qubits ($n=9$)", eimp_l9),
            (r"TQUDO qudits ($n=9$)", eimp_r9),
        ],
        y_label=r"$(E_0 - E^\star) / |E_0|$",
        x_axis_label=r"$p$",
        figsize=(6.9, 6.9),
    )
    fig.savefig(
        d_impr / "paired_n5_cq_cirq_rel_energy_vs_p.png",
        dpi=150,
    )
    plt.close(fig)

    # --- Paired n = 5: Δ P(ground) vs p (boxplots) ---
    dpl5_lists, dpr5_lists = _paired_delta_p_opt_lists_by_depth(
        xc_merged,
        depths=depths,
        output_root=root,
        bf_cache=bf_cache,
    )
    fig = _plot_paired_four_series_boxplots_vs_p(
        x_labels=x_labels,
        series=[
            ("TQUDO qubits", dpl5_lists),
            ("TQUDO qudits", dpr5_lists),
        ],
        y_label=r"$\Delta P(\mathrm{opt})$",
        x_axis_label=r"$p$",
        figsize=(8.5, 4.9),
        y_scale="symlog",
        symlog_linthresh=1e-4,
    )
    fig.savefig(
        d_popt / "paired_n5_cq_cirq_delta_popt_vs_p.png",
        dpi=150,
    )
    plt.close(fig)
