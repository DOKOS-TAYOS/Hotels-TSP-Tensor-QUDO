"""Derive paired metrics, summaries, and energy-curve aggregates from manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from data_analysis._deps import coerce_bool_scalar, require_pandas
from data_analysis.angles_stats import (
    build_angle_cohort_stats_table,
    paired_backend_angle_rows,
)
from utils.output_paths import build_output_layout


def _coerce_parse_ok(df: Any) -> Any:
    out = df.copy()
    if "parse_ok" in out.columns:
        out["parse_ok"] = out["parse_ok"].map(coerce_bool_scalar)
    return out


def _coerce_solve_ok(df: Any) -> Any:
    """Normalize ``solve_ok``; infer from ``solver_error`` for old manifests."""

    out = df.copy()
    if "solve_ok" in out.columns:
        out["solve_ok"] = out["solve_ok"].map(coerce_bool_scalar)
        return out
    if "parse_ok" not in out.columns:
        out["solve_ok"] = False
        return out
    parse_ok_series = out["parse_ok"].map(coerce_bool_scalar)
    if "solver_error" in out.columns:
        out["solve_ok"] = parse_ok_series & out["solver_error"].isna()
    else:
        out["solve_ok"] = parse_ok_series
    return out


def _load_manifest(processed: Path) -> Any:
    import pandas as pd

    pq = processed / "manifest.parquet"
    csv = processed / "manifest.csv"
    if pq.is_file():
        df = pd.read_parquet(pq)
    elif csv.is_file():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"No manifest.parquet or manifest.csv in {processed}")
    return _coerce_solve_ok(_coerce_parse_ok(df))


def _reference_bruteforce(df: Any) -> Any:
    """Ground-truth optimum: always *real* cost from ``brute_force`` + ``tqudo``.

    Objective references for ratios / first-hit curves: TQUDO BF objective for
    ``tqudo`` / ``tqudo_virtual`` rows; QUBO BF objective for ``qubo`` rows when
    a brute_force QUBO row exists for the same instance.
    """
    import pandas as pd

    bf = df[df["parse_ok"] & df["solve_ok"] & (df["solver"] == "brute_force")].copy()
    if bf.empty:
        return pd.DataFrame(
            columns=[
                "n_cities",
                "instance_key",
                "ref_real_cost",
                "ref_objective_tqudo",
                "ref_objective_qubo",
            ]
        )
    bf = bf.sort_values("path")
    bf["ref_real_cost"] = bf["best_feasible_real_cost"].where(
        bf["best_feasible_real_cost"].notna(), bf["real_cost"]
    )

    tqudo = bf[bf["formulation"] == "tqudo"]
    if tqudo.empty:
        return pd.DataFrame(
            columns=[
                "n_cities",
                "instance_key",
                "ref_real_cost",
                "ref_objective_tqudo",
                "ref_objective_qubo",
            ]
        )

    tq = tqudo.copy()
    tq["ref_objective_tqudo"] = tq["objective_value"]
    ref = tq.groupby(["n_cities", "instance_key"], as_index=False).last()
    ref = ref[
        ["n_cities", "instance_key", "ref_real_cost", "ref_objective_tqudo"]
    ]

    qubo = bf[bf["formulation"] == "qubo"]
    if not qubo.empty:
        qb = qubo.copy()
        qb["ref_objective_qubo"] = qb["objective_value"]
        qref = qb.groupby(["n_cities", "instance_key"], as_index=False).last()[
            ["n_cities", "instance_key", "ref_objective_qubo"]
        ]
        ref = ref.merge(qref, on=["n_cities", "instance_key"], how="left")
    else:
        ref["ref_objective_qubo"] = np.nan

    return ref


def build_paired_metrics(df: Any) -> Any:
    import pandas as pd

    ref = _reference_bruteforce(df)
    if ref.empty:
        out = df.copy()
        out["ref_objective_value"] = np.nan
        out["ref_real_cost"] = np.nan
        rt = pd.to_numeric(out["runtime_seconds"], errors="coerce")
        ns = pd.to_numeric(out["n_energy_steps"], errors="coerce")
        out["runtime_per_energy_step"] = np.where(
            ns.notna() & (ns > 0) & rt.notna(),
            rt / ns,
            np.nan,
        )
        return out
    merged = df.merge(
        ref,
        on=["n_cities", "instance_key"],
        how="left",
        suffixes=("", "_dup"),
    )
    ref_obj = np.full(len(merged), np.nan, dtype=np.float64)
    m_q = merged["formulation"].eq("qubo")
    m_t = merged["formulation"].isin(["tqudo", "tqudo_virtual"])
    ref_obj[m_q.to_numpy()] = merged.loc[m_q, "ref_objective_qubo"].to_numpy()
    ref_obj[m_t.to_numpy()] = merged.loc[m_t, "ref_objective_tqudo"].to_numpy()
    merged["ref_objective_value"] = ref_obj
    merged.drop(
        columns=[c for c in ("ref_objective_tqudo", "ref_objective_qubo") if c in merged.columns],
        inplace=True,
    )
    _eps = 1e-15
    ref_r = merged["ref_real_cost"].to_numpy(dtype=np.float64, copy=False)
    ref_o = merged["ref_objective_value"].to_numpy(dtype=np.float64, copy=False)
    init_e = merged["initial_energy"].to_numpy(dtype=np.float64, copy=False)
    merged["approx_ratio_real"] = np.where(
        (merged["real_cost"].notna())
        & np.isfinite(ref_r)
        & (np.abs(ref_r) > _eps),
        merged["real_cost"] / merged["ref_real_cost"],
        np.nan,
    )
    merged["approx_ratio_objective"] = np.where(
        (merged["objective_value"].notna())
        & np.isfinite(ref_o)
        & (np.abs(ref_o) > _eps),
        merged["objective_value"] / merged["ref_objective_value"],
        np.nan,
    )
    merged["energy_improvement_rel"] = np.where(
        (merged["objective_value"].notna())
        & np.isfinite(init_e)
        & (np.abs(init_e) > _eps),
        (merged["initial_energy"] - merged["objective_value"])
        / np.abs(merged["initial_energy"]),
        np.nan,
    )

    rt = pd.to_numeric(merged["runtime_seconds"], errors="coerce")
    ns = pd.to_numeric(merged["n_energy_steps"], errors="coerce")
    merged["runtime_per_energy_step"] = np.where(
        ns.notna() & (ns > 0) & rt.notna(),
        rt / ns,
        np.nan,
    )
    return merged


def build_summary_by_config(df: Any) -> Any:
    import pandas as pd

    ok = df[df["parse_ok"] & df["solve_ok"]].copy()
    if "solver" in ok.columns:
        ok = ok[ok["solver"] != "simulated_annealing"]
    if ok.empty:
        return pd.DataFrame()

    def _feas_rate(s: Any) -> float:
        return float(s.fillna(False).mean()) if len(s) else float("nan")

    def _nanmean_series(s: Any) -> float:
        x = pd.to_numeric(s, errors="coerce").dropna()
        return float(x.mean()) if len(x) else float("nan")

    def _nanmedian_series(s: Any) -> float:
        x = pd.to_numeric(s, errors="coerce").dropna()
        return float(x.median()) if len(x) else float("nan")

    rt = pd.to_numeric(ok["runtime_seconds"], errors="coerce")
    ns = pd.to_numeric(ok["n_energy_steps"], errors="coerce")
    ok["runtime_per_energy_step"] = np.where(
        ns.notna() & (ns > 0) & rt.notna(),
        rt / ns,
        np.nan,
    )

    gcols = ["n_cities", "solver", "formulation", "qaoa_depth"]
    for c in gcols:
        if c not in ok.columns:
            ok[c] = np.nan
    agg = ok.groupby(gcols, dropna=False).agg(
        n_runs=("path", "count"),
        feas_rate=("feasible", _feas_rate),
        mean_runtime=("runtime_seconds", "mean"),
        median_runtime_seconds=("runtime_seconds", _nanmedian_series),
        mean_objective=("objective_value", "mean"),
        mean_real_cost=("real_cost", "mean"),
        mean_approx_ratio_real=("approx_ratio_real", "mean"),
        mean_energy_steps=("n_energy_steps", "mean"),
        mean_configs_evaluated=("configs_evaluated", _nanmean_series),
        mean_runtime_per_energy_step=("runtime_per_energy_step", _nanmean_series),
    )
    return agg.reset_index()


def _read_energy_history(json_path: Path) -> list[float] | None:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    so = data.get("solver_output")
    if not isinstance(so, dict):
        return None
    meta = so.get("metadata")
    if not isinstance(meta, dict):
        return None
    h = meta.get("energy_history")
    if not isinstance(h, list):
        return None
    try:
        return [float(x) for x in h]
    except (TypeError, ValueError):
        return None


def read_energy_history_from_solution_json(json_path: Path) -> list[float] | None:
    """Load ``solver_output.metadata.energy_history`` from a solution JSON file."""
    return _read_energy_history(json_path)


def first_optimizer_step_reaching_min_energy(history: list[float]) -> int | None:
    """1-based optimizer-evaluation index when ``energy_history`` first hits its minimum.

    Uses mixed ``math.isclose`` tolerances for float plateaus. Returns ``None`` if history empty,
    non-finite values, or no index matches the minimum within tolerance.
    """
    if not history:
        return None
    arr = np.asarray([float(x) for x in history], dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    m = float(np.min(arr))
    atol = max(1e-12, 1e-9 * max(1.0, abs(m)))
    for i in range(int(arr.size)):
        if math.isclose(float(arr[i]), m, rel_tol=1e-9, abs_tol=atol):
            return i + 1
    return None


def aggregate_energy_curves(
    df: Any,
    output_root: Path,
    max_len_cap: int | None = None,
    *,
    include_percentiles: bool = True,
) -> Any:
    """Per-group stats of ``energy_history`` after **per-row** scaling by ``|ref_objective_value|``.

    Each instance contributes ``E(t)/|E^*_inst|``; then step-wise mean/std across instances.
    Rows without a usable BF ``ref_objective_value`` are skipped.

    ``max_len_cap``: if set, truncates all curves in the group to at most this many steps;
    if ``None`` (default), uses the longest ``energy_history`` in the group (no fixed upper bound).

    If ``include_percentiles`` is False, omits ``p25``/``p50``/``p75`` (plots only need ``mean``/``std``).
    """
    import pandas as pd

    ok = df[df["parse_ok"] & df["solve_ok"] & (df["n_energy_steps"] > 0)]
    if "solver" in ok.columns:
        ok = ok[ok["solver"] != "simulated_annealing"]
    if ok.empty:
        return pd.DataFrame()

    group_cols = ["n_cities", "solver", "formulation", "qaoa_depth"]
    if any(c not in ok.columns for c in group_cols):
        ok = ok.copy()
        for c in group_cols:
            if c not in ok.columns:
                ok[c] = np.nan
    rows_out: list[dict[str, Any]] = []

    for grp, sub in ok.groupby(group_cols, dropna=False):
        histories: list[list[float]] = []
        for _, row in sub.iterrows():
            rel = row.get("path")
            if rel is None or (isinstance(rel, float) and np.isnan(rel)):
                continue
            json_path = output_root / str(rel).strip()
            h = _read_energy_history(json_path)
            if not h:
                continue
            ref = row.get("ref_objective_value")
            if ref is None or (isinstance(ref, float) and np.isnan(ref)):
                continue
            ref_f = float(ref)
            if not np.isfinite(ref_f):
                continue
            denom = abs(ref_f)
            if denom <= 0.0:
                continue
            histories.append([float(x) / denom for x in h])
        if not histories:
            continue
        longest = max(len(h) for h in histories)
        max_len = longest if max_len_cap is None else min(int(max_len_cap), longest)
        mat = np.full((len(histories), max_len), np.nan, dtype=np.float64)
        for i, h in enumerate(histories):
            take = h[:max_len]
            mat[i, : len(take)] = take
        for step in range(max_len):
            col = mat[:, step]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            d: dict[str, Any] = {
                "step": step,
                "mean": float(np.mean(col)),
                "std": float(np.std(col, ddof=1)) if col.size > 1 else 0.0,
                "n_curves": int(col.size),
            }
            if include_percentiles:
                d["p25"] = float(np.percentile(col, 25))
                d["p50"] = float(np.median(col))
                d["p75"] = float(np.percentile(col, 75))
            if isinstance(grp, tuple):
                for k, v in zip(group_cols, grp, strict=True):
                    d[k] = v
            else:
                d[group_cols[0]] = grp
            rows_out.append(d)
    return pd.DataFrame(rows_out)


def _histogram_feasible_fraction(samples: dict[str, int], instance: dict[str, Any]) -> float | None:
    """Fraction of sample mass decoding to feasible tours (QUBO 0/1 keys or digit strings)."""
    from experiments.workflow_io import deserialize_problem_instance
    from utils.constraints import qubo_binary_to_sequence, validate_solution_constraints_tqudo

    try:
        inst = deserialize_problem_instance(instance)
    except (KeyError, TypeError, ValueError):
        return None
    n_available = inst.n_cities - 1
    total = sum(samples.values())
    if total <= 0:
        return None
    feas_mass = 0
    for key, cnt in samples.items():
        if not isinstance(key, str) or not key:
            continue
        bits = [float(int(c)) for c in key if c in "01"]
        if len(bits) != n_available * n_available:
            if len(key) == n_available and key.isdigit():
                seq_l = [int(x) for x in key]
                if validate_solution_constraints_tqudo(inst, seq_l):
                    feas_mass += cnt
            continue
        else:
            x = np.array(bits, dtype=np.float64)
            seq = qubo_binary_to_sequence(x, n_available)
        if seq is not None and validate_solution_constraints_tqudo(inst, seq):
            feas_mass += cnt
    return feas_mass / total


def enrich_sample_quality(df: Any, output_root: Path) -> Any:
    out = df.copy()
    if "solve_ok" not in out.columns:
        out = _coerce_solve_ok(out)
    fracs: list[float | None] = []
    for _, row in out.iterrows():
        frac: float | None = None
        if row.get("parse_ok") and row.get("solve_ok") and row.get("has_final_samples"):
            p = output_root / str(row["path"])
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                meta = (data.get("solver_output") or {}).get("metadata") or {}
                fs = meta.get("final_samples")
                inst = data.get("instance")
                if isinstance(fs, dict) and isinstance(inst, dict):
                    frac = _histogram_feasible_fraction(fs, inst)
            except (OSError, json.JSONDecodeError, TypeError):
                frac = None
        fracs.append(frac)
    out["final_sample_feasible_mass"] = fracs
    return out


def run_metrics(
    output_root: Path,
    sample_quality: bool,
    *,
    energy_curve_percentiles: bool = True,
) -> None:
    require_pandas(context="metrics")

    layout = build_output_layout(output_root)
    layout.processed.mkdir(parents=True, exist_ok=True)
    df = _load_manifest(layout.processed)
    paired = build_paired_metrics(df)
    if sample_quality:
        paired = enrich_sample_quality(paired, output_root)
    pq_p = layout.processed / "paired_metrics.parquet"
    paired.to_parquet(pq_p, index=False)
    paired.to_csv(layout.processed / "paired_metrics.csv", index=False)

    summary = build_summary_by_config(paired)
    summary.to_csv(layout.processed / "summary_by_config.csv", index=False)

    angle_cohort = build_angle_cohort_stats_table(paired)
    if not angle_cohort.empty:
        angle_cohort.to_parquet(layout.processed / "angle_cohort_stats.parquet", index=False)
        angle_cohort.to_csv(layout.processed / "angle_cohort_stats.csv", index=False)

    paired_angles_cq_cirq = paired_backend_angle_rows(
        paired,
        left_solver="cudaq",
        left_formulation="tqudo_virtual",
        right_solver="cirq",
        right_formulation="tqudo",
    )
    if not paired_angles_cq_cirq.empty:
        paired_angles_cq_cirq.to_parquet(
            layout.processed / "paired_angle_cudaq_tvirt_cirq_tqudo.parquet",
            index=False,
        )
        paired_angles_cq_cirq.to_csv(
            layout.processed / "paired_angle_cudaq_tvirt_cirq_tqudo.csv",
            index=False,
        )

    paired_angles_q_t = paired_backend_angle_rows(
        paired,
        left_solver="cudaq",
        left_formulation="qubo",
        right_solver="cudaq",
        right_formulation="tqudo_virtual",
    )
    if not paired_angles_q_t.empty:
        paired_angles_q_t.to_parquet(
            layout.processed / "paired_angle_cudaq_qubo_tvirt.parquet",
            index=False,
        )
        paired_angles_q_t.to_csv(
            layout.processed / "paired_angle_cudaq_qubo_tvirt.csv",
            index=False,
        )

    curves = aggregate_energy_curves(
        paired,
        output_root,
        include_percentiles=energy_curve_percentiles,
    )
    if not curves.empty:
        curves.to_parquet(layout.processed / "energy_curves_agg.parquet", index=False)
        curves.to_csv(layout.processed / "energy_curves_agg.csv", index=False)

    print(f"Wrote {pq_p}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build paired metrics from manifest.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--sample-quality",
        action="store_true",
        help="Scan JSONs for final_samples feasible mass (slower).",
    )
    parser.add_argument(
        "--no-energy-curve-percentiles",
        action="store_true",
        help="Omit p25/p50/p75 from energy_curves_agg (faster; plots use mean/std only).",
    )
    args = parser.parse_args(argv)
    run_metrics(
        args.output_root.resolve(),
        args.sample_quality,
        energy_curve_percentiles=not args.no_energy_curve_percentiles,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
