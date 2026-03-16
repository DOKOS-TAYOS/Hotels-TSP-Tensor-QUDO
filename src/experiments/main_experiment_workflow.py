"""Main experiment workflow: generate instances, solve, save results incrementally."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from instance_gen_process import (
    load_instance_config,
    load_solver_config,
    generate_random_set_instances,
    solver_config_to_run_config,
)
from instance_gen_process.config_loader import DEFAULT_CONFIG_PATH
from instance_gen_process.solver_config_loader import DEFAULT_SOLVER_CONFIG_PATH
from solvers import CudaqSolver, CirqSolver, SimulatedAnnealingSolver
from solvers.base import SolverResult
from instance_gen_process.models import ProblemInstance, InstanceConfig
from utils.output_paths import build_output_layout


def _serialize_instance(instance: ProblemInstance) -> dict[str, Any]:
    """Convert ProblemInstance to JSON-serializable dict."""
    return {
        "n_cities": instance.n_cities,
        "precedences": instance.precedences,
        "prices_hotels": instance.prices_hotels.tolist(),
        "prices_travels": instance.prices_travels.tolist(),
    }


def _serialize_instance_config(config: InstanceConfig) -> dict[str, Any]:
    """Convert InstanceConfig to JSON-serializable dict."""
    return {
        "n_cities": config.n_cities,
        "n_precedences_range": list(config.n_precedences_range),
        "prices_range_hotels": list(config.prices_range_hotels),
        "prices_range_travels": list(config.prices_range_travels),
        "seed": config.seed,
    }


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert object to JSON-serializable form."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _serialize_solver_result(result: SolverResult) -> dict[str, Any]:
    """Convert SolverResult to JSON-serializable dict."""
    return {
        "solver_name": result.solver_name,
        "objective_value": result.objective_value,
        "feasible": result.feasible,
        "runtime_seconds": result.runtime_seconds,
        "metadata": _to_json_serializable(result.metadata),
    }


def _get_solver(solver_name: str):
    """Return solver instance by name."""
    solvers = {
        "cudaq": CudaqSolver,
        "cirq": CirqSolver,
        "simulated_annealing": SimulatedAnnealingSolver,
    }
    cls = solvers.get(solver_name)
    if cls is None:
        raise ValueError(f"Unknown solver: {solver_name}. Choose from {list(solvers)}")
    return cls()


def run_workflow(
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
) -> None:
    """Run the full experiment workflow.

    Loads configs, generates instances, solves each, saves JSON after each problem.

    Args:
        instance_config_path: Path to instance config YAML. Default: config.yaml.
        solver_config_path: Path to solver config YAML. Default: solver_config.yaml.
        output_root: Root directory for output. Default: output/ relative to cwd.
    """
    instance_config = load_instance_config(instance_config_path)
    solver_config_dict = load_solver_config(solver_config_path)

    n_instances = solver_config_dict["n_instances"]
    instances = generate_random_set_instances(
        instance_config,
        n_instances,
        seed=instance_config.seed,
    )

    run_config = solver_config_to_run_config(solver_config_dict)
    solver = _get_solver(solver_config_dict["solver"])

    output_root_path = Path(output_root) if output_root else Path("output")
    layout = build_output_layout(output_root_path)
    layout.raw.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    solver_name = solver_config_dict["solver"]
    formulation = solver_config_dict["formulation"]
    restriction = solver_config_dict["restriction"]

    for i, instance in enumerate(instances):
        result = solver.solve(instance, run_config)

        solver_config_serializable: dict[str, Any] = {
            k: v for k, v in solver_config_dict.items() if k != "restriction"
        }
        solver_config_serializable["restriction"] = {
            "lambda_0": restriction.lambda_0,
            "lambda_1": restriction.lambda_1,
            "lambda_2": restriction.lambda_2,
        }

        payload: dict[str, Any] = {
            "instance": _serialize_instance(instance),
            "instance_config": _serialize_instance_config(instance_config),
            "instance_index": i,
            "solver_config": solver_config_serializable,
            "solver_output": _serialize_solver_result(result),
        }

        filename = f"exp_{timestamp}_inst_{i}_{solver_name}_{formulation}.json"
        out_path = layout.raw / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved: {out_path}")


def main() -> None:
    """CLI entry point for the experiment workflow."""
    parser = argparse.ArgumentParser(
        description="Run Hotel TSP experiment workflow: generate instances, solve, save results."
    )
    parser.add_argument(
        "--instance-config",
        type=Path,
        default=None,
        help=f"Path to instance config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--solver-config",
        type=Path,
        default=None,
        help=f"Path to solver config YAML (default: {DEFAULT_SOLVER_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Root directory for output (default: output/)",
    )
    args = parser.parse_args()

    run_workflow(
        instance_config_path=args.instance_config,
        solver_config_path=args.solver_config,
        output_root=args.output,
    )


if __name__ == "__main__":
    main()
