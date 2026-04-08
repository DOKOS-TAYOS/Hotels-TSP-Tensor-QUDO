"""Collect metrics from paired frames and on-disk solution JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data_analysis._deps import coerce_bool_scalar
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
from data_analysis.benchmark.common import _mask_qaoa_depth_eq
from data_analysis.benchmark.pairing import _dedupe_solution_rows, is_optimal_vs_ref


def float_metric_from_paired_column(col: str) -> Any:
    """Build ``value_fn`` for :func:`_collect_numeric_box_series_vs_ncities` from a DataFrame column."""

    def _fn(
        row: Any,
        _output_root: Path,
        _formulation: str,
        _bf_cache: dict[tuple[int, int], list[int] | None],
    ) -> float | None:
        v = row.get(col)
        if v is None:
            return None
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(vf):
            return None
        return vf

    return _fn


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
    """Per QAOA depth: dodged *n*, raw step counts for Cirq N-QAOA (optimal runs only)."""
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


def _approx_ratio_lists_by_depth_unpaired(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
) -> dict[int, list[float]]:
    """Per QAOA depth: list of ``approx_ratio_real`` (feasible rows only, deduped)."""
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
    """N-QAOA (Cirq native ``tqudo``) for n<9; V-QAOA (CUDA-Q ``tqudo_virtual``) for n=9 (project convention)."""
    if n_cities == 9:
        return ("cudaq", "tqudo_virtual")
    return ("cirq", "tqudo")


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
    seq = load_bruteforce_optimal_sequence(output_root, n_cities, instance_key, cache=bf_cache)
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
    seq = load_bruteforce_optimal_sequence(output_root, n_cities, instance_key, cache=bf_cache)
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
            sel = sub0[(sub0["n_cities"] == n) & (sub0["qaoa_depth"] == depth)].copy()
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
            sel = sub0[(sub0["n_cities"] == n) & (sub0["qaoa_depth"] == depth)].copy()
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
            sel = sub0[(sub0["n_cities"] == n) & (sub0["qaoa_depth"] == depth)].copy()
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
            seq = load_bruteforce_optimal_sequence(output_root, n_cities, ik, cache=bf_cache)
            if seq is None:
                continue
            key_l = histogram_key_for_formulation(seq, "tqudo_virtual", n_cities)
            key_r = histogram_key_for_formulation(seq, "tqudo", n_cities)
            init_l, fin_l = read_sample_histograms_from_solution_json(output_root / str(pl).strip())
            init_r, fin_r = read_sample_histograms_from_solution_json(output_root / str(pr).strip())
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
