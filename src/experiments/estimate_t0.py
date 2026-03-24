"""CLI tool for estimating the SA initial temperature T₀ via the Ben-Ameur method.

Arguments:
    --instance-config: Path to instance config YAML (default: src/instance_gen_process/config.yaml).
    --solver-config:   Path to solver config YAML reads formulation, restriction, seed (default: src/instance_gen_process/solver_config.yaml).
    --n-instances:     Number of random instances to sample over (default: 5).
    --chi0:            Target acceptance ratio in (0, 1) (default: 0.8).
    --n-samples:       Uphill transitions to collect per instance (default: 200).
    --epsilon:         Convergence tolerance on |chi_hat - chi_0| (default: 1e-3).
    --output:          Output directory for the JSON metadata file (default: output/T0sampling).

Usage::

    python -m experiments.estimate_t0
    python -m experiments.estimate_t0 --n-instances 3 --chi0 0.8 --n-samples 100
    python -m experiments.estimate_t0 --instance-config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from instance_gen_process import (
    generate_random_set_instances,
    load_instance_config,
    load_solver_config,
)
from instance_gen_process.models import RestrictionConfig
from solvers.simulated_annealing import T0EstimationResult, estimate_initial_temperature

from utils.experiment_serialize import serialize_instance_config, serialize_restriction_config


def _serialize_result(index: int, seed: int, result: T0EstimationResult) -> dict[str, Any]:
    return {
        "instance_index": index,
        "instance_seed": seed,
        "t0": result.t0,
        "chi_achieved": result.chi_achieved,
        "iterations": result.iterations,
        "n_samples": result.n_samples,
        "converged": result.converged,
    }


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def run_estimation(
    instance_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    n_instances: int = 5,
    chi_0: float = 0.8,
    n_samples: int = 200,
    epsilon: float = 1e-3,
    output_dir: Path | str | None = None,
) -> None:
    """Run the Ben-Ameur T₀ estimation over several random instances.

    Prints a per-instance summary and a recommended T₀ (median) to stdout,
    then saves full metadata to a JSON file.
    """
    # --- Load configs -------------------------------------------------------
    instance_config = load_instance_config(instance_config_path)
    solver_config = load_solver_config(solver_config_path)

    formulation = solver_config["formulation"]
    if formulation == "tqudo_virtual":
        formulation = "tqudo"
    restriction: RestrictionConfig = solver_config["restriction"]
    seed: int | None = solver_config["seed"]

    # --- Generate instances -------------------------------------------------
    instances = generate_random_set_instances(instance_config, n_instances, seed=seed or 42)

    # --- Header -------------------------------------------------------------
    print("\nBen-Ameur T\u2080 Estimation")
    print("\u2500" * 40)
    print(f"Formulation:  {formulation}")
    print(f"Instances:    {n_instances} (n_cities={instance_config.n_cities})")
    print(f"\u03c7\u2080 target:    {chi_0}")
    print()

    # --- Per-instance estimation --------------------------------------------
    per_instance: list[dict[str, Any]] = []
    t0_values: list[float] = []

    for i, instance in enumerate(instances):
        result = estimate_initial_temperature(
            instance,
            formulation=formulation,
            restriction=restriction,
            chi_0=chi_0,
            n_samples=n_samples,
            epsilon=epsilon,
            seed=seed,
        )
        per_instance.append(_serialize_result(i, instance.seed, result))
        t0_values.append(result.t0)

        status = "converged" if result.converged else "NOT converged"
        print(
            f"  Instance {i}: T\u2080={result.t0:<10.2f} "
            f"\u03c7={result.chi_achieved:.4f}  {status:>15s}   "
            f"({result.n_samples} samples, {result.iterations} iters)"
        )

    # --- Aggregate statistics -----------------------------------------------
    arr = np.array(t0_values)
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }
    recommended = stats["median"]

    print()
    print("\u2500" * 40)
    print(f"Recommended T\u2080 (median): {recommended:.2f}")
    print(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"Mean:  {stats['mean']:.2f}")

    # --- Save JSON ----------------------------------------------------------
    out_root = Path(output_dir) if output_dir else Path("output/T0sampling")
    out_root.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"t0_estimate_{now.strftime('%Y%m%d_%H%M%S')}.json"
    out_path = out_root / filename

    payload: dict[str, Any] = {
        "timestamp": now.isoformat(timespec="seconds"),
        "recommended_t0": recommended,
        "aggregation_method": "median",
        "statistics": stats,
        "parameters": {
            "formulation": formulation,
            "chi_0": chi_0,
            "n_samples": n_samples,
            "epsilon": epsilon,
            "seed": seed,
            "restriction": serialize_restriction_config(restriction),
        },
        "instance_config": serialize_instance_config(instance_config),
        "per_instance_results": per_instance,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nMetadata saved to: {out_path}")
    print("Update sa_t_initial in solver_config.yaml to use this value.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run :func:`run_estimation`."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the SA initial temperature T\u2080 using the Ben-Ameur method. "
            "Generates small TSP instances, samples uphill transitions, and reports "
            "the temperature that achieves a target acceptance ratio."
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
            "Path to solver config YAML — reads formulation, restriction, seed "
            "(default: src/instance_gen_process/solver_config.yaml)"
        ),
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Number of random instances to sample over (default: 5)",
    )
    parser.add_argument(
        "--chi0",
        type=float,
        default=0.8,
        help="Target acceptance ratio in (0, 1) (default: 0.8)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Uphill transitions to collect per instance (default: 200)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Convergence tolerance on |chi_hat - chi_0| (default: 1e-3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the JSON metadata file (default: output/T0sampling)",
    )
    args = parser.parse_args()

    run_estimation(
        instance_config_path=args.instance_config,
        solver_config_path=args.solver_config,
        n_instances=args.n_instances,
        chi_0=args.chi0,
        n_samples=args.n_samples,
        epsilon=args.epsilon,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
