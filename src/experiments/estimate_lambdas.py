"""CLI tool for grid-searching optimal lambda penalty parameters.

Generates small TSP instances, evaluates every combination of lambda values
on each instance using a chosen solver, and ranks combinations by feasibility
rate and mean real cost.

Arguments:
    --instance-config: Path to instance config YAML (default: src/instance_gen_process/config.yaml)
    --solver-config:   Path to solver config YAML (default: src/instance_gen_process/solver_config.yaml)
    --formulation:     Formulation to evaluate (default: qubo, choices: qubo, tqudo)
    --solver:          Solver backend (default: simulated_annealing, choices: simulated_annealing, cirq, cudaq)
    --n-instances:     Number of random instances per lambda combination (default: 5)
    --lambda-values:   Comma-separated lambda values to grid-search (default: 10,50,100,500,1000)
    --output:          Output directory for the JSON file (default: output/lambdasSampling)

Usage::

    python -m experiments.estimate_lambdas
    python -m experiments.estimate_lambdas --formulation qubo --lambda-values 10,50,100,500
    python -m experiments.estimate_lambdas --formulation tqudo --solver simulated_annealing
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

from instance_gen_process import (
    generate_random_set_instances,
    load_instance_config,
    load_solver_config,
    solver_config_to_run_config,
    validate_solver_instance_compatibility,
)
from instance_gen_process.models import InstanceConfig, ProblemInstance, RestrictionConfig
from solvers import CirqSolver, CudaqSolver, SimulatedAnnealingSolver
from solvers.base import SolverProtocol, SolverRunConfig

from experiments.json_serialize import to_json_friendly

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_instance_config(config: InstanceConfig) -> dict[str, Any]:
    return {
        "n_cities": config.n_cities,
        "n_precedences_range": list(config.n_precedences_range),
        "prices_range_hotels": list(config.prices_range_hotels),
        "prices_range_travels": list(config.prices_range_travels),
        "seed": config.seed,
    }


def _serialize_restriction(restriction: RestrictionConfig) -> dict[str, float]:
    return {
        "lambda_0": restriction.lambda_0,
        "lambda_1": restriction.lambda_1,
        "lambda_2": restriction.lambda_2,
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _get_solver(solver_name: str) -> SolverProtocol:
    """Instantiate the solver class registered for *solver_name*."""
    solvers: dict[str, type] = {
        "cudaq": CudaqSolver,
        "cirq": CirqSolver,
        "simulated_annealing": SimulatedAnnealingSolver,
    }
    cls = solvers.get(solver_name)
    if cls is None:
        raise ValueError(f"Unknown solver: {solver_name}. Choose from {list(solvers)}")
    return cls()


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
    solver: SolverProtocol,
) -> dict[str, Any]:
    """Run *solver* on every instance with the given *restriction* and aggregate."""
    run_config = dataclasses.replace(base_run_config, restriction_config=restriction)
    instance_results: list[dict[str, Any]] = []

    for idx, instance in enumerate(instances):
        try:
            result = solver.solve(instance, run_config)
            entry: dict[str, Any] = {
                "instance_index": idx,
                "instance_seed": instance.seed,
                "feasible": result.feasible,
                "objective_value": result.objective_value,
                "runtime_seconds": result.runtime_seconds,
                "best_sequence": to_json_friendly(
                    result.metadata.get("best_sequence")
                ),
                "real_cost": result.metadata.get("real_cost"),
            }
        except Exception:
            logger.warning(
                "Solver failed for instance %d with restriction %s",
                idx,
                _serialize_restriction(restriction),
                exc_info=True,
            )
            entry = {
                "instance_index": idx,
                "instance_seed": instance.seed,
                "feasible": False,
                "objective_value": None,
                "runtime_seconds": None,
                "best_sequence": None,
                "real_cost": None,
            }
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

    return {
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


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by highest feasibility rate, then lowest mean real cost."""
    return sorted(results, key=lambda r: (-r["feasibility_rate"], r["mean_real_cost"]))


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def run_lambda_sampling(
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    formulation: Literal["qubo", "tqudo"] = "qubo",
    solver_name: Literal["simulated_annealing", "cirq", "cudaq"] = "simulated_annealing",
    n_instances: int = 5,
    lambda_values: list[float] | None = None,
    output_dir: Path | str | None = None,
) -> None:
    """Run a grid search over lambda penalties and report the best combinations.

    Generates *n_instances* random problem instances, evaluates every lambda
    combination from the grid on each instance, and ranks by feasibility rate
    (primary) and mean real cost (secondary).  Prints a summary to stdout and
    saves full metadata to a JSON file.
    """
    if lambda_values is None:
        lambda_values = [10.0, 50.0, 100.0, 500.0, 1000.0]

    # --- Load configs -------------------------------------------------------
    instance_config = load_instance_config(instance_config_path)
    solver_config = load_solver_config(solver_config_path)

    original_restriction: RestrictionConfig = solver_config["restriction"]
    seed: int | None = solver_config["seed"]

    # Work on a copy to avoid mutating the loaded config dict.
    cfg = {**solver_config, "formulation": formulation, "solver": solver_name}
    validate_solver_instance_compatibility(instance_config, cfg)

    # Build base run config (restriction will be replaced per grid point).
    cfg["restriction"] = RestrictionConfig(0.0, 0.0, 0.0)
    base_run_config = solver_config_to_run_config(cfg)

    # --- Generate instances (same set for all combos) -----------------------
    instances = generate_random_set_instances(
        instance_config, n_instances, seed=seed or 42,
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
    print()

    # --- Evaluate each combo ------------------------------------------------
    solver = _get_solver(solver_name)
    all_results: list[dict[str, Any]] = []

    for i, restriction in enumerate(grid):
        combo_result = _evaluate_lambda_combo(
            restriction, instances, base_run_config, solver,
        )
        all_results.append(combo_result)

        cost_str = (
            f"{combo_result['mean_real_cost']:.1f}"
            if combo_result["mean_real_cost"] != float("inf")
            else "N/A"
        )
        print(
            f"  [{i + 1:>{len(str(len(grid)))}}/{len(grid)}] "
            f"\u03bb=({restriction.lambda_0:g}, {restriction.lambda_1:g}, "
            f"{restriction.lambda_2:g})    "
            f"feas={combo_result['feasibility_rate']:.0%}  cost={cost_str}"
        )

    # --- Rank ---------------------------------------------------------------
    ranked = _rank_results(all_results)
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
        print(
            f"  #{rank}  \u03bb0={r['lambda_0']:g}  \u03bb1={r['lambda_1']:g}  "
            f"\u03bb2={r['lambda_2']:g}    "
            f"feas={r['feasibility_rate']:.0%}  "
            f"mean_cost={cost_str}  std={std_str}"
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

    payload: dict[str, Any] = {
        "timestamp": now.isoformat(timespec="seconds"),
        "formulation": formulation,
        "solver": solver_name,
        "recommended": recommended,
        "statistics": {
            "best_feasibility_rate": best["feasibility_rate"],
            "best_mean_real_cost": best["mean_real_cost"],
        },
        "parameters": {
            "lambda_values": lambda_values,
            "n_instances": n_instances,
            "seed": seed,
            "restriction_from_config": _serialize_restriction(original_restriction),
        },
        "instance_config": _serialize_instance_config(instance_config),
        "ranked_results": to_json_friendly(ranked),
    }

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
            "feasibility rate and mean real cost."
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
        choices=["simulated_annealing", "cirq", "cudaq"],
        default="simulated_annealing",
        help="Solver backend (default: simulated_annealing)",
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
