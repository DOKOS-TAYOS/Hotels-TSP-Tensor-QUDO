"""Main experiment workflow: generate instances, solve, save results incrementally."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from instance_gen_process import (
    generate_random_set_instances,
    load_instance_config,
    load_solver_config,
    solver_config_to_run_config,
    validate_solver_instance_compatibility,
)
from instance_gen_process.config_loader import DEFAULT_CONFIG_PATH
from instance_gen_process.models import InstanceConfig, ProblemInstance
from instance_gen_process.solver_config_loader import (
    DEFAULT_SOLVER_CONFIG_PATH,
    parse_solver_config_dict,
)
from solvers import CirqSolver, CudaqSolver, SimulatedAnnealingSolver
from solvers.base import SolverProtocol, SolverResult
from config.settings import Settings, load_settings
from utils.constraints import validate_instance_constraints
from utils.cooperative_stop import (
    SolverStopRequested,
    clear_solver_stop_request,
    request_solver_stop,
)
from utils.output_paths import build_output_layout
from utils.progress import reporter

from experiments.workflow_io import (
    DEFAULT_INSTANCE_GENERATION_CONFIG_PATH,
    experiment_depth_iterations,
    instance_config_for_n_cities,
    instance_json_path,
    instances_raw_dir,
    load_instance_generation_entries,
    load_problem_instance_json,
    load_yaml_mapping,
    merge_solver_yaml_dicts,
    normalise_n_cities,
    serialize_problem_instance,
    solutions_raw_dir,
)

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent

PRESET_EXPERIMENT_YAMLS: dict[str, list[str]] = {
    "cudaq": [
        "experiment_cudaq_n5_qubo.yaml",
        "experiment_cudaq_n5_tqudo.yaml",
        "experiment_cudaq_n9_tqudo.yaml",
    ],
    "sa": [
        "experiment_sa_n5_qubo.yaml",
        "experiment_sa_n5_tqudo.yaml",
        "experiment_sa_n6_qubo.yaml",
        "experiment_sa_n6_tqudo.yaml",
    ],
    "cirq5": ["experiment_cirq_n5_tqudo.yaml"],
}


def _serialize_instance(instance: ProblemInstance) -> dict[str, Any]:
    """Convert a :class:`~instance_gen_process.models.ProblemInstance` to plain dicts/lists."""
    return serialize_problem_instance(instance)


def _serialize_instance_config(config: InstanceConfig) -> dict[str, Any]:
    """Convert :class:`~instance_gen_process.models.InstanceConfig` to JSON-friendly dict."""
    return {
        "n_cities": config.n_cities,
        "n_precedences_range": list(config.n_precedences_range),
        "prices_range_hotels": list(config.prices_range_hotels),
        "prices_range_travels": list(config.prices_range_travels),
        "seed": config.seed,
    }


def _to_json_serializable(obj: Any) -> Any:
    """Recursively normalise *obj* for JSON encoding."""
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
    """Convert :class:`~solvers.base.SolverResult` to a JSON-friendly dict."""
    return {
        "solver_name": result.solver_name,
        "objective_value": result.objective_value,
        "feasible": result.feasible,
        "runtime_seconds": result.runtime_seconds,
        "metadata": _to_json_serializable(result.metadata),
    }


def _solver_config_payload_dict(solver_config_dict: dict[str, Any]) -> dict[str, Any]:
    """Build JSON-safe solver config snapshot (expand restriction dataclass)."""
    restriction = solver_config_dict["restriction"]
    serializable: dict[str, Any] = {
        k: v for k, v in solver_config_dict.items() if k != "restriction"
    }
    serializable["restriction"] = {
        "lambda_0": restriction.lambda_0,
        "lambda_1": restriction.lambda_1,
        "lambda_2": restriction.lambda_2,
    }
    return _to_json_serializable(serializable)


def _apply_noise_kill_switch(
    run_config: Any,
    settings: Settings | None,
) -> Any:
    """Disable noise in *run_config* when settings disable noise simulation."""
    if settings is not None and not settings.enable_noise_simulation:
        if run_config.noise_config.enabled:
            silenced = dataclasses.replace(run_config.noise_config, enabled=False)
            return dataclasses.replace(run_config, noise_config=silenced)
    return run_config


def _get_solver(solver_name: str) -> SolverProtocol:
    """Instantiate the solver class registered for *solver_name*."""
    solvers = {
        "cudaq": CudaqSolver,
        "cirq": CirqSolver,
        "simulated_annealing": SimulatedAnnealingSolver,
    }
    cls = solvers.get(solver_name)
    if cls is None:
        raise ValueError(f"Unknown solver: {solver_name}. Choose from {list(solvers)}")
    return cls()


def _install_sigint_handler() -> tuple[Callable[[], bool], Callable[[], None]]:
    """Return (interrupted_predicate, clear_and_restore_placeholder).

    Uses module-level cooperative stop; restores previous SIGINT handler on clear.
    """
    clear_solver_stop_request()
    interrupted = False
    previous = signal.getsignal(signal.SIGINT)

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        request_solver_stop()
        print("\n[interrupted] stopping...", flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    def _is_interrupted() -> bool:
        return interrupted

    def _restore() -> None:
        signal.signal(signal.SIGINT, previous)

    return _is_interrupted, _restore


def run_workflow(
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
    settings: Settings | None = None,
) -> None:
    """Run the legacy experiment workflow (generate in memory, timestamped JSON names)."""
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
    run_config = _apply_noise_kill_switch(run_config, settings)

    solver = _get_solver(solver_config_dict["solver"])

    output_root_path = Path(output_root) if output_root else Path("output")
    layout = build_output_layout(output_root_path)
    layout.raw.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    solver_name = solver_config_dict["solver"]
    formulation = solver_config_dict["formulation"]

    reporter.configure(n_instances=n_instances)

    is_interrupted, restore_sigint = _install_sigint_handler()

    last_completed = -1
    n_failed = 0
    try:
        for i, instance in enumerate(instances):
            if is_interrupted():
                break
            reporter.instance_start(i)

            if not validate_instance_constraints(instance):
                logger.warning("Instance %d failed validation — skipping.", i)
                n_failed += 1
                continue

            solver_config_serializable = _solver_config_payload_dict(solver_config_dict)

            try:
                result = solver.solve(instance, run_config)
                payload: dict[str, Any] = {
                    "instance": _serialize_instance(instance),
                    "instance_config": _serialize_instance_config(instance_config),
                    "instance_index": i,
                    "solver_config": solver_config_serializable,
                    "solver_output": _serialize_solver_result(result),
                }
            except SolverStopRequested:
                break
            except Exception:
                logger.exception("Instance %d solver failed — saving error record.", i)
                n_failed += 1
                payload = {
                    "instance": _serialize_instance(instance),
                    "instance_config": _serialize_instance_config(instance_config),
                    "instance_index": i,
                    "solver_config": solver_config_serializable,
                    "solver_output": {
                        "solver_name": solver_name,
                        "error": traceback.format_exc(),
                    },
                }

            filename = f"exp_{timestamp}_inst_{i}_{solver_name}_{formulation}.json"
            out_path = layout.raw / filename
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            reporter.instance_done(i, str(out_path))
            last_completed = i
            gc.collect()
    finally:
        restore_sigint()

    if is_interrupted():
        print(
            f"[interrupted] completed {last_completed + 1} of {n_instances} instances",
            flush=True,
        )
        sys.exit(130)

    if n_failed:
        logger.warning("%d of %d instances failed.", n_failed, n_instances)


def run_generate_instances(
    instance_config_path: Path | str | None = None,
    instance_generation_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
) -> None:
    """Write ``raw/instances/n_<n_cities>/instance_<k>.json`` from generation config."""
    base_config = load_instance_config(instance_config_path)
    entries = load_instance_generation_entries(instance_generation_config_path)
    output_root_path = Path(output_root) if output_root else Path("output")

    is_interrupted, restore_sigint = _install_sigint_handler()
    try:
        for n_cities, n_instances in entries:
            if is_interrupted():
                break
            cfg = instance_config_for_n_cities(base_config, n_cities)
            instances = generate_random_set_instances(cfg, n_instances, seed=cfg.seed)
            out_dir = instances_raw_dir(output_root_path, n_cities)
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, inst in enumerate(instances, start=1):
                if is_interrupted():
                    break
                dest = out_dir / f"instance_{k}.json"
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(serialize_problem_instance(inst), f, indent=2)
                logger.info("Wrote %s", dest)
        if is_interrupted():
            sys.exit(130)
    finally:
        restore_sigint()


def run_experiment_from_yaml(
    experiment_yaml_path: Path | str,
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
    settings: Settings | None = None,
) -> None:
    """Load merged solver YAML, solve on-disk instances, write under ``raw/solutions/``."""
    base_instance = load_instance_config(instance_config_path)
    base_solver_path = Path(solver_config_path) if solver_config_path else DEFAULT_SOLVER_CONFIG_PATH
    base_solver = load_yaml_mapping(base_solver_path)
    experiment = load_yaml_mapping(experiment_yaml_path)
    merged = merge_solver_yaml_dicts(base_solver, experiment)

    if "n_cities" not in merged:
        raise ValueError(f"Experiment YAML must set n_cities: {experiment_yaml_path}")
    n_cities_list = normalise_n_cities(merged.pop("n_cities"))
    qaoa_depth_raw = merged.pop("qaoa_depth", None)
    if qaoa_depth_raw is None and "qaoa_depth" in base_solver:
        qaoa_depth_raw = base_solver["qaoa_depth"]

    solver_name = merged.get("solver", "cudaq")
    depth_iters = experiment_depth_iterations(solver_name, qaoa_depth_raw)

    n_instances = int(merged["n_instances"])
    if n_instances < 1:
        raise ValueError("n_instances must be at least 1")

    total_steps = len(n_cities_list) * len(depth_iters) * n_instances
    reporter.configure(n_instances=total_steps)

    output_root_path = Path(output_root) if output_root else Path("output")
    layout = build_output_layout(output_root_path)
    layout.raw.mkdir(parents=True, exist_ok=True)

    is_interrupted, restore_sigint = _install_sigint_handler()
    flat_i = 0
    n_failed = 0
    stop_solve = False
    try:
        for n_cities in n_cities_list:
            if stop_solve or is_interrupted():
                break
            icfg = instance_config_for_n_cities(base_instance, n_cities)
            for path_depth, run_depth in depth_iters:
                if stop_solve or is_interrupted():
                    break
                cfg_dict = {**merged, "qaoa_depth": run_depth}
                validated = parse_solver_config_dict(cfg_dict)
                validate_solver_instance_compatibility(icfg, validated)
                run_config = solver_config_to_run_config(validated)
                run_config = _apply_noise_kill_switch(run_config, settings)
                solver = _get_solver(validated["solver"])
                formulation = validated["formulation"]

                for k in range(1, n_instances + 1):
                    if is_interrupted():
                        break
                    reporter.instance_start(flat_i)
                    src = instance_json_path(output_root_path, n_cities, k)
                    if not src.is_file():
                        raise FileNotFoundError(
                            f"Missing instance file {src} (generate instances first for n_cities={n_cities})"
                        )
                    instance = load_problem_instance_json(src)
                    if not validate_instance_constraints(instance):
                        logger.warning("Instance %s failed validation — skipping.", src)
                        n_failed += 1
                        flat_i += 1
                        continue

                    solver_config_serializable = _solver_config_payload_dict(validated)
                    try:
                        result = solver.solve(instance, run_config)
                        payload = {
                            "instance": _serialize_instance(instance),
                            "instance_config": _serialize_instance_config(icfg),
                            "instance_index": k - 1,
                            "instance_source": str(src),
                            "solver_config": solver_config_serializable,
                            "solver_output": _serialize_solver_result(result),
                        }
                    except SolverStopRequested:
                        stop_solve = True
                        break
                    except Exception:
                        logger.exception("Solver failed for %s — saving error record.", src)
                        n_failed += 1
                        payload = {
                            "instance": _serialize_instance(instance),
                            "instance_config": _serialize_instance_config(icfg),
                            "instance_index": k - 1,
                            "instance_source": str(src),
                            "solver_config": solver_config_serializable,
                            "solver_output": {
                                "solver_name": solver_name,
                                "error": traceback.format_exc(),
                            },
                        }

                    out_dir = solutions_raw_dir(
                        output_root_path,
                        solver_name,
                        formulation,
                        n_cities,
                        path_depth,
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"instance_{k}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)

                    reporter.instance_done(flat_i, str(out_path))
                    flat_i += 1
                    gc.collect()

                if is_interrupted() or stop_solve:
                    break
            if is_interrupted() or stop_solve:
                break

        if is_interrupted():
            print(f"[interrupted] completed {flat_i} of {total_steps} solves", flush=True)
            sys.exit(130)

        if n_failed:
            logger.warning("%d solves failed.", n_failed)
    finally:
        restore_sigint()


def run_experiment_batch(
    experiment_yaml_paths: list[Path],
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
    settings: Settings | None = None,
) -> None:
    """Run :func:`run_experiment_from_yaml` for each path in order."""
    for p in experiment_yaml_paths:
        logger.info("Experiment YAML: %s", p)
        run_experiment_from_yaml(
            experiment_yaml_path=p,
            instance_config_path=instance_config_path,
            solver_config_path=solver_config_path,
            output_root=output_root,
            settings=settings,
        )


def main() -> None:
    """Parse CLI arguments and dispatch workflow mode."""
    parser = argparse.ArgumentParser(
        description="Hotel TSP experiment workflow: legacy, instance generation, or batched experiments."
    )
    parser.add_argument(
        "--mode",
        choices=("legacy", "generate", "cudaq", "sa", "cirq5", "experiment"),
        default="legacy",
        help="Workflow mode (default: legacy — same as before this refactor).",
    )
    parser.add_argument(
        "--instance-config",
        type=Path,
        default=None,
        help=f"Instance generation YAML (default: {DEFAULT_CONFIG_PATH} or HTSP_INSTANCE_CONFIG).",
    )
    parser.add_argument(
        "--solver-config",
        type=Path,
        default=None,
        help=f"Base solver YAML merged with experiment YAMLs (default: {DEFAULT_SOLVER_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--instance-generation-config",
        type=Path,
        default=None,
        help=f"YAML listing n_cities / n_instances blocks (default: {DEFAULT_INSTANCE_GENERATION_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--experiment-yaml",
        type=Path,
        nargs="+",
        default=None,
        help="Experiment YAML file(s); used with --mode experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root directory (default: HTSP_OUTPUT_DIR, usually output/).",
    )
    args = parser.parse_args()

    settings = load_settings()
    instance_config_path = args.instance_config or settings.instance_config_path
    output_root = args.output if args.output is not None else settings.output_dir

    if args.mode == "legacy":
        run_workflow(
            instance_config_path=instance_config_path,
            solver_config_path=args.solver_config,
            output_root=output_root,
            settings=settings,
        )
        return

    if args.mode == "generate":
        gen_cfg = args.instance_generation_config or DEFAULT_INSTANCE_GENERATION_CONFIG_PATH
        run_generate_instances(
            instance_config_path=instance_config_path,
            instance_generation_config_path=gen_cfg,
            output_root=output_root,
        )
        return

    if args.mode == "experiment":
        if not args.experiment_yaml:
            parser.error("--mode experiment requires at least one --experiment-yaml path")
        run_experiment_batch(
            experiment_yaml_paths=args.experiment_yaml,
            instance_config_path=instance_config_path,
            solver_config_path=args.solver_config,
            output_root=output_root,
            settings=settings,
        )
        return

    preset_names = PRESET_EXPERIMENT_YAMLS.get(args.mode)
    if preset_names is None:
        parser.error(f"Unknown preset mode: {args.mode!r}")
    paths = [EXPERIMENTS_DIR / name for name in preset_names]
    run_experiment_batch(
        experiment_yaml_paths=paths,
        instance_config_path=instance_config_path,
        solver_config_path=args.solver_config,
        output_root=output_root,
        settings=settings,
    )


if __name__ == "__main__":
    main()
