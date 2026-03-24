"""CLI tool for grid-searching optimal lambda penalty parameters.

Generates small TSP instances, evaluates every combination of lambda values
on each instance using a chosen solver, and ranks combinations by feasibility
rate and mean real cost (heuristic solvers), or — with ``solver=brute_force`` —
by feasibility of the global minimum and mean gap to the combinatorial minimum
real cost over TQUDO-feasible sequences (reference independent of λ).

**Brute force:** requires an instance config that passes
``validate_solver_instance_compatibility`` for ``solver=brute_force`` (e.g. QUBO
needs small ``n_cities`` so ``(n_cities-1)² ≤ 30`` binary variables; TQUDO needs
``n_cities - 1 ≤ 8``). The default ``config.yaml`` with ``n_cities: 9`` fits
TQUDO brute force but not QUBO brute force.

**CPU parallel instances:** for ``cirq``, ``simulated_annealing``, and
``brute_force``, different random instances in a λ combo are solved in parallel
(``multiprocessing`` spawn) when ``cpu_max_parallel_instances`` in the solver YAML
(or ``HTSP_CPU_MAX_PARALLEL_INSTANCES``) is greater than 1 — same mechanism as
the on-disk experiment workflow. CUDA-Q stays sequential here (use its own GPU
parallel settings in the main workflow).

Arguments:
    --instance-config: Path to instance config YAML (default: src/instance_gen_process/config.yaml)
    --solver-config:   Path to solver config YAML (default: src/instance_gen_process/solver_config.yaml)
    --formulation:     Formulation to evaluate (default: qubo, choices: qubo, tqudo)
    --solver:          Solver backend (default: simulated_annealing; includes brute_force)
    --n-instances:     Number of random instances per lambda combination (default: 5)
    --lambda-values:   Comma-separated lambda values to grid-search (default: 10,50,100,500,1000)
    --output:          Output directory for the JSON file (default: output/lambdasSampling)

Usage::

    python -m experiments.estimate_lambdas
    python -m experiments.estimate_lambdas --formulation qubo --lambda-values 10,50,100,500
    python -m experiments.estimate_lambdas --formulation tqudo --solver simulated_annealing
    python -m experiments.estimate_lambdas --solver brute_force --formulation tqudo --lambda-values 100,500
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing as mp
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np

from experiments.parallel_solve_batch import (
    CPU_PARALLEL_ENV,
    EXPERIMENT_CUDA_WORKER_ENV,
    resolve_cpu_max_parallel_instances,
)
from instance_gen_process import (
    DEFAULT_SOLVER_CONFIG_PATH,
    generate_random_set_instances,
    load_instance_config,
    load_solver_config,
    solver_config_to_run_config,
    validate_solver_instance_compatibility,
)
from instance_gen_process.models import ProblemInstance, RestrictionConfig
from solvers import CirqSolver, CudaqSolver, SimulatedAnnealingSolver
from solvers.base import SolverProtocol, SolverRunConfig
from solvers.brute_force import BruteForceSolver

from utils.constraints import validate_solution_constraints_tqudo
from utils.costs import calculate_real_cost
from utils.costs_batch import unpack_tqudo_sequences
from utils.experiment_serialize import (
    serialize_instance_config,
    serialize_restriction_config,
)
from utils.json_serialize import to_json_friendly
from utils.yaml_tools import read_solver_yaml_as_mapping

logger = logging.getLogger(__name__)

_PARALLEL_SOLVER_YAML_KEYS = ("cudaq_max_parallel_instances", "cpu_max_parallel_instances")


def _parallel_fields_from_solver_yaml(path: Path | str | None) -> dict[str, Any]:
    """Keys not returned by :func:`load_solver_config` but needed for parallel resolution."""
    p = Path(path) if path is not None else DEFAULT_SOLVER_CONFIG_PATH
    raw = read_solver_yaml_as_mapping(p)
    return {k: raw[k] for k in _PARALLEL_SOLVER_YAML_KEYS if k in raw}

# Recovery of combinatorial optimal real cost (brute-force ranking only).
_GAP_RECOVERY_EPS = 1e-6
_REFERENCE_CHUNK = 8192

# Same backends as on-disk CPU parallel batch (not CUDA-Q).
_CPU_PARALLEL_SOLVERS = frozenset({"brute_force", "simulated_annealing", "cirq"})


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _reference_min_feasible_real_cost(
    instance: ProblemInstance,
) -> tuple[float | None, list[int] | None]:
    """Minimum ``calculate_real_cost`` over TQUDO-feasible sequences (independent of λ)."""
    n_available = instance.n_cities - 1
    cardinal = n_available**n_available
    best_cost: float | None = None
    best_seq: list[int] | None = None

    for i0 in range(0, cardinal, _REFERENCE_CHUNK):
        i1 = min(i0 + _REFERENCE_CHUNK, cardinal)
        i_vals = np.arange(i0, i1, dtype=np.int64)
        seqs = unpack_tqudo_sequences(i_vals, n_available)
        for j in range(i1 - i0):
            seq_list = seqs[j].tolist()
            if validate_solution_constraints_tqudo(instance, seq_list):
                c = calculate_real_cost(instance, seq_list)
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_seq = seq_list

    return (best_cost, best_seq)


def _build_reference_table(instances: list[ProblemInstance]) -> list[dict[str, Any]]:
    """Per-instance combinatorial optimum real cost (for brute-force λ ranking)."""
    table: list[dict[str, Any]] = []
    for inst in instances:
        opt, seq = _reference_min_feasible_real_cost(inst)
        table.append(
            {
                "has_feasible_reference": opt is not None,
                "reference_optimal_real_cost": opt,
                "reference_optimal_sequence": to_json_friendly(seq) if seq is not None else None,
            },
        )
    return table


def _get_solver(solver_name: str) -> SolverProtocol:
    """Instantiate the solver class registered for *solver_name*."""
    solvers: dict[str, type] = {
        "cudaq": CudaqSolver,
        "cirq": CirqSolver,
        "simulated_annealing": SimulatedAnnealingSolver,
        "brute_force": BruteForceSolver,
    }
    cls = solvers.get(solver_name)
    if cls is None:
        raise ValueError(f"Unknown solver: {solver_name}. Choose from {list(solvers)}")
    return cls()


def _estimate_lambdas_solve_one_worker(
    job: tuple[int, str, ProblemInstance, SolverRunConfig],
) -> tuple[int, dict[str, Any]]:
    """Top-level for spawn: run one solver on one instance in a child process."""
    idx, solver_name, instance, run_config = job
    os.environ[EXPERIMENT_CUDA_WORKER_ENV] = "1"
    solver = _get_solver(solver_name)
    try:
        result = solver.solve(instance, run_config)
        return idx, {
            "ok": True,
            "feasible": result.feasible,
            "objective_value": result.objective_value,
            "runtime_seconds": result.runtime_seconds,
            "best_sequence": result.metadata.get("best_sequence"),
            "real_cost": result.metadata.get("real_cost"),
        }
    except Exception:
        logger.exception(
            "estimate_lambdas worker failed for instance index %s",
            idx,
        )
        return idx, {"ok": False}


def _solve_instances_for_combo(
    restriction: RestrictionConfig,
    instances: list[ProblemInstance],
    base_run_config: SolverRunConfig,
    solver_name: str,
    max_parallel_instances: int,
) -> list[dict[str, Any]]:
    """Run the solver on each instance; parallelize across instances for CPU backends."""
    run_config = dataclasses.replace(base_run_config, restriction_config=restriction)
    n = len(instances)
    if n == 0:
        return []

    use_parallel = (
        max_parallel_instances > 1
        and n > 1
        and solver_name in _CPU_PARALLEL_SOLVERS
    )

    if not use_parallel:
        solver = _get_solver(solver_name)
        raw: list[dict[str, Any]] = []
        for idx in range(n):
            try:
                result = solver.solve(instances[idx], run_config)
                raw.append(
                    {
                        "ok": True,
                        "feasible": result.feasible,
                        "objective_value": result.objective_value,
                        "runtime_seconds": result.runtime_seconds,
                        "best_sequence": result.metadata.get("best_sequence"),
                        "real_cost": result.metadata.get("real_cost"),
                    },
                )
            except Exception:
                logger.warning(
                    "Solver failed for instance %d with restriction %s",
                    idx,
                    serialize_restriction_config(restriction),
                    exc_info=True,
                )
                raw.append({"ok": False})
        return raw

    workers = min(max_parallel_instances, n)
    jobs = [(idx, solver_name, instances[idx], run_config) for idx in range(n)]
    ctx = mp.get_context("spawn")
    raw_by_idx: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        future_to_idx = {
            executor.submit(_estimate_lambdas_solve_one_worker, job): job[0]
            for job in jobs
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                idx2, payload = fut.result()
                if idx2 != idx:
                    logger.error("Worker index mismatch: expected %s got %s", idx, idx2)
                raw_by_idx[idx] = payload
            except Exception:
                logger.exception("estimate_lambdas future failed for instance index %s", idx)
                raw_by_idx[idx] = {"ok": False}
    return [raw_by_idx[i] for i in range(n)]


def _generate_lambda_grid(
    formulation: Literal["qubo", "tqudo"],
    lambda_values: list[float],
) -> list[RestrictionConfig]:
    """Build the grid of :class:`RestrictionConfig` to evaluate.

    For ``qubo`` all three lambdas are varied; for ``tqudo`` only
    ``lambda_1`` and ``lambda_2`` are varied (``lambda_0`` is fixed at 0).
    """
    if formulation == "qubo":
        return [
            RestrictionConfig(lambda_0=l0, lambda_1=l1, lambda_2=l2)
            for l0, l1, l2 in product(lambda_values, repeat=3)
        ]
    if formulation == "tqudo":
        return [
            RestrictionConfig(lambda_0=0.0, lambda_1=l1, lambda_2=l2)
            for l1, l2 in product(lambda_values, repeat=2)
        ]
    raise ValueError(f"formulation must be 'qubo' or 'tqudo', got {formulation!r}")


def _evaluate_lambda_combo(
    restriction: RestrictionConfig,
    instances: list[ProblemInstance],
    base_run_config: SolverRunConfig,
    solver_name: str,
    *,
    max_parallel_instances: int = 1,
    reference_table: list[dict[str, Any]] | None = None,
    use_brute_force_metrics: bool = False,
) -> dict[str, Any]:
    """Run *solver* on every instance with the given *restriction* and aggregate."""
    raw_rows = _solve_instances_for_combo(
        restriction,
        instances,
        base_run_config,
        solver_name,
        max_parallel_instances,
    )
    instance_results: list[dict[str, Any]] = []

    for idx, instance in enumerate(instances):
        d = raw_rows[idx]
        if d.get("ok"):
            entry = {
                "instance_index": idx,
                "instance_seed": instance.seed,
                "feasible": d["feasible"],
                "objective_value": d["objective_value"],
                "runtime_seconds": d["runtime_seconds"],
                "best_sequence": to_json_friendly(d.get("best_sequence")),
                "real_cost": d.get("real_cost"),
            }
        else:
            entry = {
                "instance_index": idx,
                "instance_seed": instance.seed,
                "feasible": False,
                "objective_value": None,
                "runtime_seconds": None,
                "best_sequence": None,
                "real_cost": None,
            }

        if use_brute_force_metrics and reference_table is not None:
            ref = reference_table[idx]
            entry["gap_to_reference"] = None
            if ref["has_feasible_reference"] and ref["reference_optimal_real_cost"] is not None:
                ref_cost = float(ref["reference_optimal_real_cost"])
                if entry["feasible"] and entry["real_cost"] is not None:
                    gap = float(entry["real_cost"]) - ref_cost
                    entry["gap_to_reference"] = gap
                else:
                    entry["gap_to_reference"] = float("inf")

        instance_results.append(entry)

    n_total = len(instance_results)
    feasible_results = [r for r in instance_results if r["feasible"]]
    n_feasible = len(feasible_results)
    feasibility_rate = n_feasible / n_total if n_total > 0 else 0.0

    feasible_costs = [
        r["real_cost"] for r in feasible_results if r["real_cost"] is not None
    ]
    mean_real_cost = (
        statistics.mean(feasible_costs) if feasible_costs else float("inf")
    )
    std_real_cost = (
        statistics.stdev(feasible_costs) if len(feasible_costs) > 1 else 0.0
    )

    out: dict[str, Any] = {
        "lambda_0": restriction.lambda_0,
        "lambda_1": restriction.lambda_1,
        "lambda_2": restriction.lambda_2,
        "feasibility_rate": feasibility_rate,
        "mean_real_cost": mean_real_cost,
        "std_real_cost": std_real_cost,
        "n_feasible": n_feasible,
        "n_total": n_total,
        "instance_results": instance_results,
    }

    if use_brute_force_metrics and reference_table is not None:
        gaps_eligible: list[float] = []
        n_ref_eligible = 0
        n_recovered = 0
        for idx, ref in enumerate(reference_table):
            if not ref["has_feasible_reference"] or ref["reference_optimal_real_cost"] is None:
                continue
            n_ref_eligible += 1
            g = instance_results[idx].get("gap_to_reference")
            if g is None:
                continue
            gaps_eligible.append(float(g))
            if (
                instance_results[idx]["feasible"]
                and instance_results[idx]["real_cost"] is not None
                and math.isfinite(g)
                and abs(g) <= _GAP_RECOVERY_EPS
            ):
                n_recovered += 1

        mean_gap = statistics.mean(gaps_eligible) if gaps_eligible else float("inf")
        finite_gaps = [g for g in gaps_eligible if math.isfinite(g)]
        std_gap = statistics.stdev(finite_gaps) if len(finite_gaps) > 1 else 0.0
        optimal_recovery_rate = (
            n_recovered / n_ref_eligible if n_ref_eligible > 0 else 0.0
        )

        out["mean_gap_to_reference"] = mean_gap
        out["std_gap_to_reference"] = std_gap
        out["optimal_recovery_rate"] = optimal_recovery_rate
        out["n_reference_eligible"] = n_ref_eligible

    return out


def _rank_results(
    results: list[dict[str, Any]],
    *,
    ranking_mode: Literal["default", "brute_force_gap"] = "default",
) -> list[dict[str, Any]]:
    """Sort by highest feasibility rate, then secondary key (mean real cost or mean gap)."""
    if ranking_mode == "default":
        return sorted(
            results,
            key=lambda r: (-r["feasibility_rate"], r["mean_real_cost"]),
        )
    return sorted(
        results,
        key=lambda r: (
            -r["feasibility_rate"],
            r.get("mean_gap_to_reference", float("inf")),
        ),
    )


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def run_lambda_sampling(
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    formulation: Literal["qubo", "tqudo"] = "qubo",
    solver_name: Literal[
        "simulated_annealing",
        "cirq",
        "cudaq",
        "brute_force",
    ] = "simulated_annealing",
    n_instances: int = 5,
    lambda_values: list[float] | None = None,
    output_dir: Path | str | None = None,
) -> None:
    """Run a grid search over lambda penalties and report the best combinations.

    Generates *n_instances* random problem instances, evaluates every lambda
    combination from the grid on each instance, and ranks by feasibility rate
    (primary) and mean real cost (secondary) for heuristic solvers, or mean gap
    to the combinatorial optimum (secondary) when ``solver_name`` is
    ``brute_force``.  Prints a summary to stdout and saves full metadata to a
    JSON file.
    """
    if lambda_values is None:
        lambda_values = [10.0, 50.0, 100.0, 500.0, 1000.0]

    # --- Load configs -------------------------------------------------------
    instance_config = load_instance_config(instance_config_path)
    solver_config = load_solver_config(solver_config_path)
    parallel_yaml = _parallel_fields_from_solver_yaml(solver_config_path)

    original_restriction: RestrictionConfig = solver_config["restriction"]
    seed: int | None = solver_config["seed"]

    # Work on a copy to avoid mutating the loaded config dict.
    cfg = {
        **solver_config,
        **parallel_yaml,
        "formulation": formulation,
        "solver": solver_name,
    }
    validate_solver_instance_compatibility(instance_config, cfg)

    cpu_parallel_effective = 1
    if solver_name in _CPU_PARALLEL_SOLVERS:
        cpu_parallel_effective = max(
            1,
            min(resolve_cpu_max_parallel_instances(cfg), n_instances),
        )

    # Build base run config (restriction will be replaced per grid point).
    cfg["restriction"] = RestrictionConfig(0.0, 0.0, 0.0)
    base_run_config = solver_config_to_run_config(cfg)

    # --- Generate instances (same set for all combos) -----------------------
    instances = generate_random_set_instances(
        instance_config, n_instances, seed=seed or 42,
    )

    use_brute_force_metrics = solver_name == "brute_force"
    reference_table: list[dict[str, Any]] | None = None
    if use_brute_force_metrics:
        reference_table = _build_reference_table(instances)
        for i, row in enumerate(reference_table):
            if not row["has_feasible_reference"]:
                logger.warning(
                    "Instance %d has no TQUDO-feasible sequence; "
                    "excluded from reference gap statistics",
                    i,
                )

    # --- Build grid ---------------------------------------------------------
    grid = _generate_lambda_grid(formulation, lambda_values)

    # --- Header -------------------------------------------------------------
    print("\nLambda Penalty Grid Search")
    print("\u2500" * 40)
    print(f"Formulation:  {formulation}")
    print(f"Solver:       {solver_name}")
    print(f"Instances:    {n_instances} (n_cities={instance_config.n_cities})")
    print(f"Grid size:    {len(grid)} combinations")
    if solver_name in _CPU_PARALLEL_SOLVERS:
        print(
            f"CPU parallel: {cpu_parallel_effective} worker(s) per λ combo "
            f"(cpu_max_parallel_instances / {CPU_PARALLEL_ENV})",
        )
    if use_brute_force_metrics:
        print("Ranking:      feasibility rate, then mean gap to combinatorial min real cost")
    print()

    # --- Evaluate each combo ------------------------------------------------
    all_results: list[dict[str, Any]] = []

    for i, restriction in enumerate(grid):
        combo_result = _evaluate_lambda_combo(
            restriction,
            instances,
            base_run_config,
            solver_name,
            max_parallel_instances=cpu_parallel_effective,
            reference_table=reference_table,
            use_brute_force_metrics=use_brute_force_metrics,
        )
        all_results.append(combo_result)

        cost_str = (
            f"{combo_result['mean_real_cost']:.1f}"
            if combo_result["mean_real_cost"] != float("inf")
            else "N/A"
        )
        line = (
            f"  [{i + 1:>{len(str(len(grid)))}}/{len(grid)}] "
            f"\u03bb=({restriction.lambda_0:g}, {restriction.lambda_1:g}, "
            f"{restriction.lambda_2:g})    "
            f"feas={combo_result['feasibility_rate']:.0%}  cost={cost_str}"
        )
        if use_brute_force_metrics:
            mg = combo_result.get("mean_gap_to_reference", float("inf"))
            mg_str = f"{mg:.4g}" if math.isfinite(mg) else "inf"
            rec = combo_result.get("optimal_recovery_rate", 0.0)
            line += f"  mean_gap={mg_str}  recover={rec:.0%}"
        print(line)

    # --- Rank ---------------------------------------------------------------
    ranking_mode: Literal["default", "brute_force_gap"] = (
        "brute_force_gap" if use_brute_force_metrics else "default"
    )
    ranked = _rank_results(all_results, ranking_mode=ranking_mode)
    best = ranked[0]

    # --- Console summary ----------------------------------------------------
    print()
    print("\u2500" * 40)
    print("Top 5 Lambda Combinations:")
    for rank, r in enumerate(ranked[:5], start=1):
        cost_str = (
            f"{r['mean_real_cost']:.1f}" if r["mean_real_cost"] != float("inf") else "N/A"
        )
        std_str = f"{r['std_real_cost']:.1f}" if r["std_real_cost"] > 0 else "0.0"
        extra = ""
        if use_brute_force_metrics:
            mg = r.get("mean_gap_to_reference", float("inf"))
            mg_str = f"{mg:.4g}" if math.isfinite(mg) else "inf"
            rec = r.get("optimal_recovery_rate", 0.0)
            extra = f"  mean_gap={mg_str}  recover={rec:.0%}"
        print(
            f"  #{rank}  \u03bb0={r['lambda_0']:g}  \u03bb1={r['lambda_1']:g}  "
            f"\u03bb2={r['lambda_2']:g}    "
            f"feas={r['feasibility_rate']:.0%}  "
            f"mean_cost={cost_str}  std={std_str}{extra}"
        )

    recommended = {
        "lambda_0": best["lambda_0"],
        "lambda_1": best["lambda_1"],
        "lambda_2": best["lambda_2"],
    }
    print()
    print(
        f"Recommended: \u03bb0={recommended['lambda_0']:g}  "
        f"\u03bb1={recommended['lambda_1']:g}  "
        f"\u03bb2={recommended['lambda_2']:g}"
    )

    # --- Save JSON ----------------------------------------------------------
    out_root = Path(output_dir) if output_dir else Path("output/lambdasSampling")
    out_root.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"lambda_grid_{now.strftime('%Y%m%d_%H%M%S')}.json"
    out_path = out_root / filename

    statistics: dict[str, Any] = {
        "best_feasibility_rate": best["feasibility_rate"],
        "best_mean_real_cost": best["mean_real_cost"],
    }
    if use_brute_force_metrics:
        statistics["best_mean_gap_to_reference"] = best.get("mean_gap_to_reference")
        statistics["best_optimal_recovery_rate"] = best.get("optimal_recovery_rate")

    payload: dict[str, Any] = {
        "timestamp": now.isoformat(timespec="seconds"),
        "formulation": formulation,
        "solver": solver_name,
        "ranking_mode": ranking_mode,
        "recommended": recommended,
        "statistics": statistics,
        "parameters": {
            "lambda_values": lambda_values,
            "n_instances": n_instances,
            "seed": seed,
            "restriction_from_config": serialize_restriction_config(original_restriction),
            "cpu_max_parallel_instances_effective": (
                cpu_parallel_effective if solver_name in _CPU_PARALLEL_SOLVERS else 1
            ),
        },
        "instance_config": serialize_instance_config(instance_config),
        "ranked_results": to_json_friendly(ranked),
    }
    if use_brute_force_metrics and reference_table is not None:
        payload["reference_per_instance"] = to_json_friendly(reference_table)
        payload["gap_recovery_epsilon"] = _GAP_RECOVERY_EPS

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nMetadata saved to: {out_path}")
    print("Update restriction in solver_config.yaml to use these values.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run :func:`run_lambda_sampling`."""
    parser = argparse.ArgumentParser(
        description=(
            "Grid search over lambda penalty parameters for QUBO/TQUDO formulations. "
            "Evaluates every combination on small random instances and ranks by "
            "feasibility rate and mean real cost (heuristic solvers), or — with "
            "brute_force — mean gap to combinatorial minimum real cost."
        ),
    )
    parser.add_argument(
        "--instance-config",
        type=Path,
        default=Path("src/instance_gen_process/config.yaml"),
        help="Path to instance config YAML (default: src/instance_gen_process/config.yaml)",
    )
    parser.add_argument(
        "--solver-config",
        type=Path,
        default=Path("src/instance_gen_process/solver_config.yaml"),
        help=(
            "Path to solver config YAML — reads SA/QAOA params, seed; "
            "lambdas are overridden by the grid (default: src/instance_gen_process/solver_config.yaml)"
        ),
    )
    parser.add_argument(
        "--formulation",
        choices=["qubo", "tqudo"],
        default="qubo",
        help="Formulation to evaluate (default: qubo)",
    )
    parser.add_argument(
        "--solver",
        choices=["simulated_annealing", "cirq", "cudaq", "brute_force"],
        default="simulated_annealing",
        help=(
            "Solver backend (default: simulated_annealing). "
            "Use brute_force for exact global minima; requires an instance size "
            "compatible with brute-force limits (see module docstring)."
        ),
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Number of random instances per lambda combination (default: 5)",
    )
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="10,50,100,500,1000",
        help="Comma-separated lambda values to grid-search (default: 10,50,100,500,1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the JSON file (default: output/lambdasSampling)",
    )
    args = parser.parse_args()

    lambda_values = [float(v.strip()) for v in args.lambda_values.split(",")]

    run_lambda_sampling(
        instance_config_path=args.instance_config,
        solver_config_path=args.solver_config,
        formulation=args.formulation,
        solver_name=args.solver,
        n_instances=args.n_instances,
        lambda_values=lambda_values,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
