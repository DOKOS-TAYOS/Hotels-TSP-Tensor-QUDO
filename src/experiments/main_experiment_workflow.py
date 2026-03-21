"""Main experiment workflow: generate instances, solve, save results incrementally."""

from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from instance_gen_process import (
    load_instance_config,
    load_solver_config,
    generate_random_set_instances,
    solver_config_to_run_config,
    validate_solver_instance_compatibility,
)
from instance_gen_process.config_loader import DEFAULT_CONFIG_PATH
from instance_gen_process.solver_config_loader import DEFAULT_SOLVER_CONFIG_PATH
from solvers import CudaqSolver, CirqSolver, SimulatedAnnealingSolver
from solvers.base import SolverResult
from config.settings import Settings, load_settings
from instance_gen_process.models import ProblemInstance, InstanceConfig
from utils.output_paths import build_output_layout
from utils.progress import reporter


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
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_json_serializable(dataclasses.asdict(obj))
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
    settings: "Settings | None" = None,
) -> None:
    """Run the full experiment workflow.

    Loads configs, generates instances, solves each, saves JSON after each problem.

    Args:
        instance_config_path: Path to instance config YAML. Default: config.yaml.
        solver_config_path: Path to solver config YAML. Default: solver_config.yaml.
        output_root: Root directory for output. Default: output/ relative to cwd.
        settings: Optional runtime settings.  When provided and
            ``enable_noise_simulation`` is False, noise is forcibly disabled
            regardless of the YAML config (environment kill-switch).
    """
    instance_config = load_instance_config(instance_config_path)
    solver_config_dict = load_solver_config(solver_config_path)
    validate_solver_instance_compatibility(instance_config, solver_config_dict)

    n_instances = solver_config_dict["n_instances"]
    instances = generate_random_set_instances(
        instance_config,
        n_instances,
        seed=instance_config.seed,
    )

    run_config = solver_config_to_run_config(solver_config_dict)

    # --- Environment kill-switch (option B): if HTSP_ENABLE_NOISE_SIMULATION
    # is explicitly set to false in the environment / .env file, override the
    # YAML noise.enabled flag to guarantee noise is off.
    if settings is not None and not settings.enable_noise_simulation:
        if run_config.noise_config.enabled:
            silenced = dataclasses.replace(run_config.noise_config, enabled=False)
            run_config = dataclasses.replace(run_config, noise_config=silenced)

    solver = _get_solver(solver_config_dict["solver"])

    output_root_path = Path(output_root) if output_root else Path("output")
    layout = build_output_layout(output_root_path)
    layout.raw.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    solver_name = solver_config_dict["solver"]
    formulation = solver_config_dict["formulation"]
    restriction = solver_config_dict["restriction"]

    reporter.configure(n_instances=n_instances)

    _interrupted = False

    def _handle_sigint(sig: int, frame: object) -> None:
        nonlocal _interrupted
        _interrupted = True
        print("\n[interrupted] finishing current instance then stopping...", flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    for i, instance in enumerate(instances):
        if _interrupted:
            break
        reporter.instance_start(i)
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
            "solver_config": _to_json_serializable(solver_config_serializable),
            "solver_output": _serialize_solver_result(result),
        }

        filename = f"exp_{timestamp}_inst_{i}_{solver_name}_{formulation}.json"
        out_path = layout.raw / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        reporter.instance_done(i, str(out_path))

    if _interrupted:
        print("[interrupted] stopped after instance", i, flush=True)
        sys.exit(130)


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

    settings = load_settings()
    instance_config_path = args.instance_config or settings.instance_config_path
    run_workflow(
        instance_config_path=instance_config_path,
        solver_config_path=args.solver_config,
        output_root=args.output,
        settings=settings,
    )


if __name__ == "__main__":
    main()
