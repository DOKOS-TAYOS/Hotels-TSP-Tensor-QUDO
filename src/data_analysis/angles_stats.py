"""QAOA angle similarity: cohort pairwise cosine and paired backend comparisons."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

# Max QAOA depth supported for flattened std_* columns in CSV exports.
_MAX_ANGLE_DEPTH_FLAT: int = 8


def concat_normalize_angles(
    gamma: list[float],
    beta: list[float],
) -> np.ndarray:
    """Concatenate gamma and beta, return L2-unit vector (shape ``(2p,)``).

    If the norm is zero, returns a vector of NaNs of the expected length.
    """
    p = len(gamma)
    if p != len(beta) or p == 0:
        return np.full(2 * max(p, len(beta), 1), np.nan, dtype=np.float64)
    v = np.asarray(gamma + beta, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 0.0 or not np.isfinite(n):
        return np.full(v.shape[0], np.nan, dtype=np.float64)
    return v / n


def gamma_beta_from_row(row: Any) -> tuple[list[float] | None, list[float] | None]:
    """Read QAOA angles from a manifest/paired row (Parquet lists or JSON strings)."""
    g = row.get("oa_gamma") if hasattr(row, "get") else getattr(row, "oa_gamma", None)
    b = row.get("oa_beta") if hasattr(row, "get") else getattr(row, "oa_beta", None)
    if isinstance(g, str) and g.strip():
        try:
            g = json.loads(g)
        except json.JSONDecodeError:
            g = None
    if isinstance(b, str) and b.strip():
        try:
            b = json.loads(b)
        except json.JSONDecodeError:
            b = None
    if not isinstance(g, list) or not isinstance(b, list):
        gj = row.get("oa_gamma_json") if hasattr(row, "get") else None
        bj = row.get("oa_beta_json") if hasattr(row, "get") else None
        if isinstance(gj, str) and gj.strip():
            try:
                g = json.loads(gj)
            except json.JSONDecodeError:
                g = None
        if isinstance(bj, str) and bj.strip():
            try:
                b = json.loads(bj)
            except json.JSONDecodeError:
                b = None
    if not isinstance(g, list) or not isinstance(b, list):
        return None, None
    try:
        gf = [float(x) for x in g]
        bf = [float(x) for x in b]
    except (TypeError, ValueError):
        return None, None
    if len(gf) != len(bf) or len(gf) == 0:
        return None, None
    if not np.all(np.isfinite(gf)) or not np.all(np.isfinite(bf)):
        return None, None
    return gf, bf


def cohort_angle_stats(subdf: Any) -> dict[str, Any]:
    """Aggregate angle similarity for rows in one `(n_cities, solver, formulation, p)` cohort."""
    triples: list[tuple[np.ndarray, list[float], list[float]]] = []
    for _, row in subdf.iterrows():
        g, b = gamma_beta_from_row(row)
        if g is None:
            continue
        v = concat_normalize_angles(g, b)
        if not np.all(np.isfinite(v)):
            continue
        triples.append((v, g, b))

    n_runs = int(len(subdf))
    n_raw = len(triples)
    out: dict[str, Any] = {
        "n_runs": n_runs,
        "n_runs_with_angles": n_raw,
        "angle_vector_dim_used": np.nan,
        "n_runs_angles_dim_consistent": 0,
        "mean_pairwise_cosine": np.nan,
        "std_pairwise_cosine": np.nan,
    }

    def _flat_std_nan() -> None:
        out["std_gamma_json"] = None
        out["std_beta_json"] = None
        for k in range(_MAX_ANGLE_DEPTH_FLAT):
            out[f"std_gamma_{k}"] = np.nan
            out[f"std_beta_{k}"] = np.nan

    if n_raw == 0:
        _flat_std_nan()
        return out

    lengths = [t[0].size for t in triples]
    mode_len = Counter(lengths).most_common(1)[0][0]
    filt = [t for t in triples if t[0].size == mode_len]
    n_with = len(filt)
    out["angle_vector_dim_used"] = float(mode_len)
    out["n_runs_angles_dim_consistent"] = n_with

    vecs = [t[0] for t in filt]
    gammas = [t[1] for t in filt]
    betas = [t[2] for t in filt]

    if n_with == 0:
        _flat_std_nan()
        return out

    if n_with == 1:
        out["std_gamma_json"] = json.dumps([0.0] * len(gammas[0]))
        out["std_beta_json"] = json.dumps([0.0] * len(betas[0]))
        for k in range(_MAX_ANGLE_DEPTH_FLAT):
            p0 = len(gammas[0])
            out[f"std_gamma_{k}"] = 0.0 if k < p0 else np.nan
            out[f"std_beta_{k}"] = 0.0 if k < p0 else np.nan
        return out

    mat = np.stack(vecs, axis=0)
    cosines: list[float] = []
    for i in range(n_with):
        for j in range(i + 1, n_with):
            c = float(np.dot(mat[i], mat[j]))
            cosines.append(c)
    carr = np.asarray(cosines, dtype=np.float64)
    out["mean_pairwise_cosine"] = float(np.mean(carr))
    out["std_pairwise_cosine"] = float(np.std(carr, ddof=1)) if len(cosines) > 1 else 0.0

    g_arr = np.asarray(gammas, dtype=np.float64)
    b_arr = np.asarray(betas, dtype=np.float64)
    std_g = np.std(g_arr, axis=0, ddof=1).tolist()
    std_b = np.std(b_arr, axis=0, ddof=1).tolist()
    out["std_gamma_json"] = json.dumps([float(x) for x in std_g])
    out["std_beta_json"] = json.dumps([float(x) for x in std_b])
    for k in range(_MAX_ANGLE_DEPTH_FLAT):
        out[f"std_gamma_{k}"] = float(std_g[k]) if k < len(std_g) else np.nan
        out[f"std_beta_{k}"] = float(std_b[k]) if k < len(std_b) else np.nan
    return out


def _merged_row_side_dict(row: Any, side: str, keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = row[k]
    suf = f"_{side}"
    for name in row.index:
        if isinstance(name, str) and name.endswith(suf):
            out[name[: -len(suf)]] = row[name]
    return out


def paired_backend_angle_rows(
    paired_df: Any,
    *,
    left_solver: str,
    left_formulation: str,
    right_solver: str,
    right_formulation: str,
) -> Any:
    """Inner join on `(n_cities, instance_key, qaoa_depth)`; cosine and L2 between norm vectors."""
    import pandas as pd

    left = paired_df[
        (paired_df["solver"] == left_solver)
        & (paired_df["formulation"] == left_formulation)
        & paired_df["parse_ok"]
        & paired_df["solve_ok"]
    ].copy()
    right = paired_df[
        (paired_df["solver"] == right_solver)
        & (paired_df["formulation"] == right_formulation)
        & paired_df["parse_ok"]
        & paired_df["solve_ok"]
    ].copy()
    keys = ("n_cities", "instance_key", "qaoa_depth")
    for k in keys:
        if k not in left.columns or k not in right.columns:
            return pd.DataFrame()
    keyl = list(keys)
    if "path" in left.columns:
        left = left.sort_values("path").drop_duplicates(subset=keyl, keep="last")
    else:
        left = left.drop_duplicates(subset=keyl, keep="last")
    if "path" in right.columns:
        right = right.sort_values("path").drop_duplicates(subset=keyl, keep="last")
    else:
        right = right.drop_duplicates(subset=keyl, keep="last")
    merged = left.merge(
        right,
        on=list(keys),
        how="inner",
        suffixes=("_left", "_right"),
    )
    rows_out: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        row_l = _merged_row_side_dict(row, "left", keys)
        row_r = _merged_row_side_dict(row, "right", keys)
        gl, bl = gamma_beta_from_row(row_l)
        gr, br = gamma_beta_from_row(row_r)
        if gl is None or gr is None:
            continue
        if len(gl) != len(gr):
            continue
        vl = concat_normalize_angles(gl, bl)
        vr = concat_normalize_angles(gr, br)
        if not np.all(np.isfinite(vl)) or not np.all(np.isfinite(vr)):
            continue
        cosine = float(np.dot(vl, vr))
        l2_delta = float(np.linalg.norm(vl - vr))
        rows_out.append(
            {
                "n_cities": row["n_cities"],
                "instance_key": row["instance_key"],
                "qaoa_depth": row["qaoa_depth"],
                "left_solver": left_solver,
                "left_formulation": left_formulation,
                "right_solver": right_solver,
                "right_formulation": right_formulation,
                "cosine_pair": cosine,
                "l2_delta": l2_delta,
            }
        )
    return pd.DataFrame(rows_out)


def build_angle_cohort_stats_table(paired: Any) -> Any:
    """One row per `(n_cities, solver, formulation, qaoa_depth)` with angle dispersion stats."""
    import pandas as pd

    gcols = ["n_cities", "solver", "formulation", "qaoa_depth"]
    ok = paired[paired["parse_ok"] & paired["solve_ok"]].copy()
    if ok.empty:
        return pd.DataFrame()
    if "solver" in ok.columns:
        ok = ok[ok["solver"] != "brute_force"]
    if ok.empty:
        return pd.DataFrame()
    for c in gcols:
        if c not in ok.columns:
            ok[c] = np.nan
    out_rows: list[dict[str, Any]] = []
    for grp, sub in ok.groupby(gcols, dropna=False):
        stats = cohort_angle_stats(sub)
        if isinstance(grp, tuple):
            rec = {k: v for k, v in zip(gcols, grp, strict=True)}
        else:
            rec = {gcols[0]: grp}
        rec.update(stats)
        out_rows.append(rec)
    return pd.DataFrame(out_rows)
