"""Paired comparative dashboards and mean approximation-ratio plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from data_analysis.metrics import (
    first_optimizer_step_reaching_min_energy,
    read_energy_history_from_solution_json,
)

_RTOL_REAL = 1e-6
_ATOL_REAL = 1e-8


def _row_bool(x: Any) -> bool:
    if x is True:
        return True
    if x is False:
        return False
    return str(x).lower() == "true"


def is_optimal_vs_ref(
    real_cost: float | None,
    ref_real_cost: float | None,
    feasible: Any,
) -> bool:
    if ref_real_cost is None or (isinstance(ref_real_cost, float) and math.isnan(ref_real_cost)):
        return False
    if real_cost is None or (isinstance(real_cost, float) and math.isnan(real_cost)):
        return False
    if not _row_bool(feasible):
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

    l_feas = merged["feasible_left"].map(_row_bool)
    r_feas = merged["feasible_right"].map(_row_bool)
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
    ax00.set_ylabel("Instances")
    ax00.legend(loc="upper right", fontsize=8)
    ax00.set_xlabel(x_axis_label)

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
    ax01.set_ylabel("% (both feasible)")
    ax01.set_ylim(0, 105)
    ax01.legend(fontsize=7)
    ax01.set_xlabel(x_axis_label)

    ax10 = axes[1, 0]
    olf = [float(s["only_left_feasible_pct"]) for s in stats_list]
    orf = [float(s["only_right_feasible_pct"]) for s in stats_list]
    ax10.bar(x - ww, olf, ww, label=f"{label_left} only", color="#1f77b4")
    ax10.bar(x + ww, orf, ww, label=f"{label_right} only", color="#ff7f0e")
    ax10.set_xticks(x)
    ax10.set_xticklabels(x_labels)
    ax10.set_ylabel("% paired")
    ax10.set_ylim(0, 105)
    ax10.legend(fontsize=8)
    ax10.set_xlabel(x_axis_label)

    ax11 = axes[1, 1]
    olo = [float(s["only_left_optimal_pct"]) for s in stats_list]
    oro = [float(s["only_right_optimal_pct"]) for s in stats_list]
    ax11.bar(x - ww, olo, ww, label=f"{label_left} only", color="#2ca02c")
    ax11.bar(x + ww, oro, ww, label=f"{label_right} only", color="#d62728")
    ax11.set_xticks(x)
    ax11.set_xticklabels(x_labels)
    ax11.set_ylabel("% paired")
    ax11.set_ylim(0, 105)
    ax11.legend(fontsize=8)
    ax11.set_xlabel(x_axis_label)

    fig.tight_layout()
    return fig


def _merged_both_optimal_mask(merged: Any) -> Any:
    """Rows where left and right are optimal vs BF TQUDO real-cost reference."""
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
    return l_opt & r_opt


def _plot_comparison_opt_steps_bars(
    *,
    x_labels: list[str],
    means_left: list[float],
    stds_left: list[float],
    means_right: list[float],
    stds_right: list[float],
    counts: list[int],
    label_left: str,
    label_right: str,
    x_axis_label: str,
) -> Any:
    """Plot grouped bars of mean steps to trace minimum energy (paired rows, optimal on both sides)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(x_labels), dtype=np.float64)
    w = 0.36
    ml = np.array([0.0 if (v != v) else float(v) for v in means_left], dtype=np.float64)
    mr = np.array([0.0 if (v != v) else float(v) for v in means_right], dtype=np.float64)
    el = np.array(
        [0.0 if c == 0 or (s != s) else float(s) for s, c in zip(stds_left, counts, strict=True)],
        dtype=np.float64,
    )
    er = np.array(
        [0.0 if c == 0 or (s != s) else float(s) for s, c in zip(stds_right, counts, strict=True)],
        dtype=np.float64,
    )
    ax.bar(x - w / 2, ml, w, yerr=el, capsize=3, label=label_left, color="#1f77b4")
    ax.bar(x + w / 2, mr, w, yerr=er, capsize=3, label=label_right, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Steps (mean ± σ)")
    ax.legend(fontsize=8)
    ax.set_xlabel(x_axis_label)
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


def _collect_paired_opt_steps_by_depth(
    merged: Any,
    *,
    depths: tuple[int, ...],
    output_root: Path,
) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    """Per QAOA depth: mean/std of steps for pairs optimal on both sides (JSON read at plot time)."""
    means_l: list[float] = []
    stds_l: list[float] = []
    means_r: list[float] = []
    stds_r: list[float] = []
    counts: list[int] = []
    if merged.empty or "path_left" not in merged.columns:
        for _ in depths:
            counts.append(0)
            means_l.append(float("nan"))
            stds_l.append(float("nan"))
            means_r.append(float("nan"))
            stds_r.append(float("nan"))
        return means_l, stds_l, means_r, stds_r, counts
    for d in depths:
        sub = merged[merged["qaoa_depth"].astype(float).astype(int) == int(d)]
        sub = sub.loc[_merged_both_optimal_mask(sub)].copy()
        steps_l: list[float] = []
        steps_r: list[float] = []
        for _, row in sub.iterrows():
            sl = _opt_steps_from_rel_path(output_root, row.get("path_left"))
            sr = _opt_steps_from_rel_path(output_root, row.get("path_right"))
            if sl is not None and sr is not None:
                steps_l.append(sl)
                steps_r.append(sr)
        n = len(steps_l)
        counts.append(n)
        if n == 0:
            means_l.append(float("nan"))
            stds_l.append(float("nan"))
            means_r.append(float("nan"))
            stds_r.append(float("nan"))
            continue
        a = np.asarray(steps_l, dtype=np.float64)
        b = np.asarray(steps_r, dtype=np.float64)
        means_l.append(float(a.mean()))
        stds_l.append(float(a.std(ddof=1)) if n > 1 else 0.0)
        means_r.append(float(b.mean()))
        stds_r.append(float(b.std(ddof=1)) if n > 1 else 0.0)
    return means_l, stds_l, means_r, stds_r, counts


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
    feas = sub["feasible"].map(_row_bool)
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


def _solver_form_tqudo_by_n_cities(n_cities: int) -> tuple[str, str]:
    """Cirq native TQUDO for n<9; CUDA-Q TQUDO virtual for n=9 (project convention)."""
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
    feas = sub["feasible"].map(_row_bool)
    ar = sub["approx_ratio_real"]
    sub = sub.loc[feas & ar.notna(), "approx_ratio_real"]
    if sub.empty:
        return None
    mean = float(sub.mean())
    std = float(sub.std(ddof=1)) if len(sub) > 1 else 0.0
    if std != std:  # NaN
        std = 0.0
    return mean, std


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


def _plot_mean_approx_ratio_vs_ncities(
    series: list[tuple[str, list[float], list[float], list[float]]],
    *,
    n_tick_vals: list[int],
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Any:
    """``series``: (label, x, y, yerr) with x already including depth dodge."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    markers = ("o", "s", "^", "D", "v", "P", "*", "X", "h", "8")
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    for i, (label, xs, means, stds) in enumerate(series):
        if not xs:
            continue
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
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ρ = 1")
    ax.set_xticks([float(n) for n in n_tick_vals])
    ax.set_xticklabels([str(n) for n in n_tick_vals])
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\rho$ (mean ± σ)")
    ax.legend(fontsize=7 if len(series) > 4 else 8)
    fig.tight_layout()
    return fig


def _plot_mean_approx_ratio_points(
    series: list[tuple[str, list[int], list[float], list[float]]],
    *,
    figsize: tuple[float, float] = (7.5, 4.8),
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
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ρ = 1")
    all_depths = sorted({d for _, ds, _, _ in series for d in ds})
    if all_depths:
        ax.set_xticks(all_depths)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\rho$ (mean ± σ)")
    ax.legend(fontsize=7 if len(series) > 4 else 8)
    fig.tight_layout()
    return fig


def run_benchmark_plots(paired: Any, output_root: Path, images_dir: Path) -> None:
    """Write CUDA-Q / cross-backend dashboard and mean approximation-ratio figures."""
    import matplotlib.pyplot as plt

    images_dir.mkdir(parents=True, exist_ok=True)

    # --- CUDA-Q: QUBO vs TQUDO virtual by QAOA depth (n_cities = 5) ---
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
        label_right="TQUDO virt.",
        x_axis_label=r"$p$",
    )
    fig.savefig(
        images_dir / "comparison_cudaq_qubo_vs_tqudo_virtual_by_qaoa_depth.png",
        dpi=150,
    )
    plt.close(fig)

    # --- CUDA-Q TQUDO virtual vs Cirq native TQUDO by QAOA depth ---
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
        label_left="CQ virt.",
        label_right="Cirq TQUDO",
        x_axis_label=r"$p$",
    )
    fig.savefig(
        images_dir / "comparison_cudaq_tqudo_virtual_vs_cirq_tqudo_by_qaoa_depth.png",
        dpi=150,
    )
    plt.close(fig)

    # --- Mean approximation ratio (unpaired): single figure, multiple series ---
    d_q, y_q, e_q = _mean_approx_ratio_by_depth_unpaired(
        paired, solver="cudaq", formulation="qubo", n_cities=5
    )
    d_tv5, y_tv5, e_tv5 = _mean_approx_ratio_by_depth_unpaired(
        paired, solver="cudaq", formulation="tqudo_virtual", n_cities=5
    )
    d_ci5, y_ci5, e_ci5 = _mean_approx_ratio_by_depth_unpaired(
        paired, solver="cirq", formulation="tqudo", n_cities=5
    )
    fig = _plot_mean_approx_ratio_points(
        [
            ("QUBO", d_q, y_q, e_q),
            ("TQUDO virt.", d_tv5, y_tv5, e_tv5),
            ("Cirq TQUDO", d_ci5, y_ci5, e_ci5),
        ],
    )
    fig.savefig(
        images_dir
        / "comparison_mean_approx_ratio_cudaq_qubo_cudaq_tvirt_cirq_tqudo_n5_by_qaoa_depth.png",
        dpi=150,
    )
    plt.close(fig)

    n_multi = [5, 6, 7, 8, 9]
    vs_n_series = _mean_approx_ratio_series_vs_ncities_by_depth(paired, n_values=n_multi)
    fig = _plot_mean_approx_ratio_vs_ncities(
        vs_n_series,
        n_tick_vals=n_multi,
        figsize=(8.5, 5),
    )
    fig.savefig(
        images_dir
        / "comparison_mean_approx_ratio_cirq_tqudo_n5_n8_cudaq_tvirt_n9_by_ncities.png",
        dpi=150,
    )
    plt.close(fig)

    # --- Paired comparison: steps to trace min energy (both optimal; JSON read at plot time) ---
    ml, sl, mr, sr, cnt = _collect_paired_opt_steps_by_depth(
        cq_merged, depths=depths, output_root=output_root.resolve()
    )
    fig = _plot_comparison_opt_steps_bars(
        x_labels=x_labels,
        means_left=ml,
        stds_left=sl,
        means_right=mr,
        stds_right=sr,
        counts=cnt,
        label_left="QUBO",
        label_right="TQUDO virt.",
        x_axis_label=r"$p$",
    )
    fig.savefig(
        images_dir
        / "comparison_cudaq_qubo_vs_tqudo_virtual_opt_steps_both_optimal_by_qaoa_depth.png",
        dpi=150,
    )
    plt.close(fig)

    ml2, sl2, mr2, sr2, cnt2 = _collect_paired_opt_steps_by_depth(
        xc_merged, depths=depths, output_root=output_root.resolve()
    )
    fig = _plot_comparison_opt_steps_bars(
        x_labels=x_labels,
        means_left=ml2,
        stds_left=sl2,
        means_right=mr2,
        stds_right=sr2,
        counts=cnt2,
        label_left="CQ virt.",
        label_right="Cirq TQUDO",
        x_axis_label=r"$p$",
    )
    fig.savefig(
        images_dir
        / "comparison_cudaq_tqudo_virtual_vs_cirq_tqudo_opt_steps_both_optimal_by_qaoa_depth.png",
        dpi=150,
    )
    plt.close(fig)
