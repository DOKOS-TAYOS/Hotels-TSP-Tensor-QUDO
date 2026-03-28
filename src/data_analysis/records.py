"""Parse experiment JSON payloads into flat manifest rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_analysis.instance_features import instance_features_from_json_dict


def manifest_empty_schema_row() -> dict[str, Any]:
    """Single row template so an empty manifest still has predictable columns."""
    return {
        "path": "",
        "layout": "empty",
        "parse_ok": False,
        "solve_ok": False,
        "solver": None,
        "formulation": None,
        "n_cities": None,
        "instance_key": None,
        "qaoa_depth": None,
        "n_cities_json": None,
        "solver_config_solver": None,
        "solver_config_formulation": None,
        "seed": None,
        "noise_enabled": False,
        "instance_index": None,
        "feasible": None,
        "objective_value": None,
        "runtime_seconds": None,
        "real_cost": None,
        "n_energy_steps": 0,
        "has_final_samples": False,
        "has_initial_samples": False,
        "initial_energy": None,
        "best_feasible_objective_value": None,
        "best_feasible_real_cost": None,
        "configs_evaluated": None,
        "solver_error": None,
        "error_message": None,
        "inst_n_precedences": None,
        "inst_precedence_density": None,
        "inst_prices_hotels_mean": None,
        "inst_prices_hotels_std": None,
        "inst_prices_hotels_range": None,
        "inst_prices_travels_pos_mean": None,
        "inst_prices_travels_pos_std": None,
        "oa_gamma": None,
        "oa_beta": None,
        "oa_gamma_json": None,
        "oa_beta_json": None,
    }


def _parse_solutions_subpath(parts: tuple[str, ...]) -> dict[str, Any] | None:
    """Parse ``solver/formulation/n_{n}/[depth]/instance_{k}.json`` under raw/solutions.

    Also accepts ``solver/solver/formulation/n_{n}/...`` (duplicate solver folder).
    """
    if len(parts) < 4:
        return None
    if len(parts) >= 5 and parts[0] == parts[1] and not parts[2].startswith("n_"):
        parts = (parts[0],) + parts[2:]
    if len(parts) < 4:
        return None
    solver, formulation, n_raw = parts[0], parts[1], parts[2]
    if not n_raw.startswith("n_"):
        return None
    try:
        n_cities = int(n_raw[2:])
    except ValueError:
        return None
    tail = parts[3:]
    if not tail:
        return None
    last = tail[-1]
    if not last.startswith("instance_") or not last.endswith(".json"):
        return None
    try:
        instance_key = int(last[9:-5])
    except ValueError:
        return None
    middle = tail[:-1]
    qaoa_depth: int | None
    if not middle:
        qaoa_depth = None
    elif len(middle) == 1 and middle[0].isdigit():
        qaoa_depth = int(middle[0])
    else:
        return None
    return {
        "solver": solver,
        "formulation": formulation,
        "n_cities": n_cities,
        "instance_key": instance_key,
        "qaoa_depth": qaoa_depth,
    }


def path_context(path: Path, output_root: Path) -> dict[str, Any]:
    """Infer layout and path-derived fields (may be partial for unknown layouts)."""
    try:
        rel = path.relative_to(output_root)
    except ValueError:
        return {"layout": "unknown", "path": str(path)}
    parts = rel.parts
    ctx: dict[str, Any] = {"layout": "unknown", "path": str(rel)}
    if (
        len(parts) >= 6
        and parts[0] == "raw"
        and parts[1] == "solutions"
    ):
        sub = _parse_solutions_subpath(parts[2:])
        if sub is not None:
            ctx["layout"] = "disk"
            ctx.update(sub)
    return ctx


def _safe_len_history(meta: dict[str, Any]) -> int:
    h = meta.get("energy_history")
    if isinstance(h, list):
        return len(h)
    return 0


def _coerce_float_list(x: Any) -> list[float] | None:
    if not isinstance(x, list):
        return None
    out: list[float] = []
    for v in x:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return out


def _optimal_angles_row(
    meta: dict[str, Any],
    path_qaoa_depth: int | None,
    solver_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Parse ``metadata.optimal_angles`` into manifest columns."""
    empty = {
        "oa_gamma": None,
        "oa_beta": None,
        "oa_gamma_json": None,
        "oa_beta_json": None,
    }
    oa = meta.get("optimal_angles")
    if not isinstance(oa, dict):
        return empty
    gamma = _coerce_float_list(oa.get("gamma"))
    beta = _coerce_float_list(oa.get("beta"))
    if gamma is None or beta is None or len(gamma) != len(beta) or len(gamma) == 0:
        return empty

    exp_depth = path_qaoa_depth
    if exp_depth is None and solver_config is not None:
        qd = solver_config.get("qaoa_depth")
        if qd is not None:
            try:
                exp_depth = int(qd)
            except (TypeError, ValueError):
                exp_depth = None
    if exp_depth is not None and len(gamma) != exp_depth:
        return empty

    return {
        "oa_gamma": gamma,
        "oa_beta": beta,
        "oa_gamma_json": json.dumps(gamma),
        "oa_beta_json": json.dumps(beta),
    }


def json_row(path: Path, output_root: Path) -> dict[str, Any]:
    """Load JSON at *path* and return one manifest row.

    ``parse_ok`` is True when the file is valid JSON with a top-level object.
    ``solve_ok`` is True when ``solver_output`` is present and has no ``error``
    key (a normal solver result). Failed solves stored by the workflow still
    have ``parse_ok`` True but ``solve_ok`` False.
    """
    ctx = path_context(path, output_root)
    row: dict[str, Any] = {
        "path": ctx.get("path", str(path)),
        "layout": ctx.get("layout", "unknown"),
        "parse_ok": False,
        "solve_ok": False,
    }
    for k in (
        "solver",
        "formulation",
        "n_cities",
        "instance_key",
        "qaoa_depth",
    ):
        if k in ctx:
            row[k] = ctx[k]

    try:
        with open(path, encoding="utf-8") as f:
            data: Any = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        row["error_message"] = str(exc)
        row["solve_ok"] = False
        return row

    if not isinstance(data, dict):
        row["error_message"] = "top-level JSON is not an object"
        row["solve_ok"] = False
        return row

    row["parse_ok"] = True
    inst = data.get("instance")
    if isinstance(inst, dict):
        row.update(instance_features_from_json_dict(inst))
        if "n_cities" in inst:
            row["n_cities_json"] = int(inst["n_cities"])

    sc = data.get("solver_config")
    if isinstance(sc, dict):
        row["solver_config_solver"] = sc.get("solver")
        row["solver_config_formulation"] = sc.get("formulation")
        row["seed"] = sc.get("seed")
        noise = sc.get("noise")
        if isinstance(noise, dict):
            row["noise_enabled"] = bool(noise.get("enabled", False))
        else:
            row["noise_enabled"] = False
        if row.get("solver") is None:
            row["solver"] = sc.get("solver")
        if row.get("formulation") is None:
            row["formulation"] = sc.get("formulation")

    if "instance_index" in data:
        row["instance_index"] = int(data["instance_index"])

    so = data.get("solver_output")
    if not isinstance(so, dict):
        row["solver_error"] = "missing solver_output"
        row.update(
            {
                "oa_gamma": None,
                "oa_beta": None,
                "oa_gamma_json": None,
                "oa_beta_json": None,
            }
        )
        return row

    if "error" in so:
        row["solver_error"] = str(so.get("error", ""))[:2000]
        row.update(
            {
                "oa_gamma": None,
                "oa_beta": None,
                "oa_gamma_json": None,
                "oa_beta_json": None,
            }
        )
        return row

    row["feasible"] = so.get("feasible")
    row["objective_value"] = so.get("objective_value")
    row["runtime_seconds"] = so.get("runtime_seconds")
    meta = so.get("metadata")
    if isinstance(meta, dict):
        row["real_cost"] = meta.get("real_cost")
        row["n_energy_steps"] = _safe_len_history(meta)
        row["has_final_samples"] = "final_samples" in meta
        row["has_initial_samples"] = "initial_samples" in meta
        row["initial_energy"] = meta.get("initial_energy")
        row["best_feasible_objective_value"] = meta.get("best_feasible_objective_value")
        row["best_feasible_real_cost"] = meta.get("best_feasible_real_cost")
        row["configs_evaluated"] = meta.get("configs_evaluated")
        cfg_for_angles = sc if isinstance(sc, dict) else None
        row.update(_optimal_angles_row(meta, row.get("qaoa_depth"), cfg_for_angles))
    else:
        row["n_energy_steps"] = 0
        row["has_final_samples"] = False
        row["has_initial_samples"] = False
        row.update(
            {
                "oa_gamma": None,
                "oa_beta": None,
                "oa_gamma_json": None,
                "oa_beta_json": None,
            }
        )

    if row.get("n_cities") is None and row.get("n_cities_json") is not None:
        row["n_cities"] = row["n_cities_json"]

    if row.get("instance_key") is None and row.get("instance_index") is not None:
        row["instance_key"] = int(row["instance_index"]) + 1

    row["solve_ok"] = True
    return row
