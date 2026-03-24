"""Tests for experiments.estimate_lambdas (brute-force reference and ranking)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from instance_gen_process.models import ProblemInstance, RestrictionConfig
from experiments.estimate_lambdas import (
    _build_reference_table,
    _evaluate_lambda_combo,
    _rank_results,
    _reference_min_feasible_real_cost,
    run_lambda_sampling,
)
from solvers.base import SolverRunConfig
from utils.constraints import validate_solution_constraints_tqudo
from utils.costs import calculate_real_cost
from utils.costs_batch import unpack_tqudo_sequences


def _tiny_instance() -> ProblemInstance:
    """n_cities=3 => two visitable cities; small enumeration."""
    n = 3
    pt = np.ones((n, n, n), dtype=np.float64) * 5.0
    return ProblemInstance(
        n_cities=n,
        precedences=(),
        prices_hotels=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        prices_travels=pt,
        seed=0,
    )


def test_reference_min_feasible_real_cost_is_minimum_over_feasible_sequences() -> None:
    instance = _tiny_instance()
    ref_cost, ref_seq = _reference_min_feasible_real_cost(instance)
    assert ref_cost is not None and ref_seq is not None
    assert validate_solution_constraints_tqudo(instance, ref_seq)
    n_available = instance.n_cities - 1
    cardinal = n_available**n_available
    manual_best: float | None = None
    for i in range(cardinal):
        seq = unpack_tqudo_sequences(np.array([i], dtype=np.int64), n_available)[0].tolist()
        if validate_solution_constraints_tqudo(instance, seq):
            c = calculate_real_cost(instance, seq)
            if manual_best is None or c < manual_best:
                manual_best = c
    assert manual_best is not None
    assert ref_cost == pytest.approx(manual_best)


def test_rank_results_brute_force_gap_orders_by_mean_gap() -> None:
    results = [
        {
            "feasibility_rate": 1.0,
            "mean_real_cost": 10.0,
            "mean_gap_to_reference": 2.0,
        },
        {
            "feasibility_rate": 1.0,
            "mean_real_cost": 5.0,
            "mean_gap_to_reference": 0.5,
        },
    ]
    ranked = _rank_results(results, ranking_mode="brute_force_gap")
    assert ranked[0]["mean_gap_to_reference"] == 0.5


def test_evaluate_lambda_combo_parallel_two_instances_smoke() -> None:
    """ProcessPoolExecutor path for two instances (CPU parallel)."""
    instance = _tiny_instance()
    restriction = RestrictionConfig(10.0, 10.0, 100.0)
    run = SolverRunConfig(
        formulation="tqudo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )
    ref = _build_reference_table([instance, instance])
    combo = _evaluate_lambda_combo(
        restriction,
        [instance, instance],
        run,
        "brute_force",
        max_parallel_instances=2,
        reference_table=ref,
        use_brute_force_metrics=True,
    )
    assert combo["n_total"] == 2
    assert "mean_gap_to_reference" in combo


def test_evaluate_lambda_combo_brute_force_includes_gap_fields() -> None:
    instance = _tiny_instance()
    ref = _build_reference_table([instance])
    restriction = RestrictionConfig(10.0, 10.0, 100.0)
    run = SolverRunConfig(
        formulation="tqudo",
        restriction_config=restriction,
        brute_force_max_assignments_tqudo=1000,
        brute_force_max_assignments_qubo=1000,
    )
    combo = _evaluate_lambda_combo(
        restriction,
        [instance],
        run,
        "brute_force",
        max_parallel_instances=1,
        reference_table=ref,
        use_brute_force_metrics=True,
    )
    assert "mean_gap_to_reference" in combo
    assert "optimal_recovery_rate" in combo
    assert combo["n_reference_eligible"] == 1


def test_run_lambda_sampling_brute_force_json_payload(tmp_path: Path) -> None:
    instance_yaml = tmp_path / "instance.yaml"
    instance_yaml.write_text(
        "\n".join(
            [
                "n_cities: 3",
                "n_precedences_range: [0, 0]",
                "prices_range_hotels: [1.0, 10.0]",
                "prices_range_travels: [1.0, 10.0]",
                "seed: 0",
            ],
        ),
        encoding="utf-8",
    )
    solver_yaml = tmp_path / "solver.yaml"
    solver_yaml.write_text(
        "\n".join(
            [
                "n_instances: 1",
                "solver: brute_force",
                "formulation: tqudo",
                "optimizer: COBYLA",
                "restriction:",
                "  lambda_0: 100.0",
                "  lambda_1: 100.0",
                "  lambda_2: 100.0",
                "qaoa_depth: 1",
                "qaoa_max_iter: 100",
                "seed: 0",
            ],
        ),
        encoding="utf-8",
    )
    run_lambda_sampling(
        instance_config_path=instance_yaml,
        solver_config_path=solver_yaml,
        formulation="tqudo",
        solver_name="brute_force",
        n_instances=1,
        lambda_values=[100.0],
        output_dir=tmp_path,
    )
    out_files = list(tmp_path.glob("lambda_grid_*.json"))
    assert len(out_files) == 1
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert payload["solver"] == "brute_force"
    assert payload["ranking_mode"] == "brute_force_gap"
    assert "reference_per_instance" in payload
    assert payload["statistics"].get("best_mean_gap_to_reference") is not None
    assert "ranked_results" in payload
