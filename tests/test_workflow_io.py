"""Unit tests for experiment workflow YAML merge and instance JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.workflow_io import (
    DEFAULT_INSTANCE_GENERATION_CONFIG_PATH,
    deserialize_problem_instance,
    experiment_depth_iterations,
    instance_json_path,
    instances_raw_dir,
    load_instance_generation_entries,
    merge_solver_yaml_dicts,
    normalise_n_cities,
    serialize_problem_instance,
    solutions_raw_dir,
    solutions_solver_root,
)
from instance_gen_process.models import ProblemInstance
from instance_gen_process.solver_config_loader import parse_solver_config_dict


def test_merge_solver_yaml_dicts_restriction_and_top_level() -> None:
    base = {
        "solver": "cudaq",
        "n_instances": 1,
        "restriction": {"lambda_0": 1.0, "lambda_1": 2.0, "lambda_2": 3.0},
        "qaoa_depth": 1,
    }
    override = {
        "solver": "cirq",
        "restriction": {"lambda_0": 9.0},
        "formulation": "tqudo",
    }
    merged = merge_solver_yaml_dicts(base, override)
    assert merged["solver"] == "cirq"
    assert merged["formulation"] == "tqudo"
    assert merged["n_instances"] == 1
    assert merged["restriction"]["lambda_0"] == 9.0
    assert merged["restriction"]["lambda_1"] == 2.0
    assert merged["restriction"]["lambda_2"] == 3.0


def test_serialize_deserialize_problem_instance_roundtrip() -> None:
    inst = ProblemInstance(
        n_cities=3,
        precedences=((0, 1), (1, 2)),
        prices_hotels=np.array([[1.0, 2.0], [3.0, 4.0]]),
        prices_travels=np.ones((3, 3, 3)),
        seed=12345,
    )
    data = serialize_problem_instance(inst)
    raw = json.loads(json.dumps(data))
    back = deserialize_problem_instance(raw)
    assert back.n_cities == inst.n_cities
    assert back.precedences == inst.precedences
    assert back.seed == inst.seed
    np.testing.assert_array_equal(back.prices_hotels, inst.prices_hotels)
    np.testing.assert_array_equal(back.prices_travels, inst.prices_travels)


def test_instance_and_solution_paths() -> None:
    root = Path("/tmp/out")
    assert instances_raw_dir(root, 5) == Path("/tmp/out/raw/instances/n_5")
    assert instance_json_path(root, 5, 3) == Path("/tmp/out/raw/instances/n_5/instance_3.json")
    assert solutions_solver_root(root, "cudaq") == Path("/tmp/out/raw/solutions/cudaq")
    assert solutions_raw_dir(root, "cudaq", "qubo", 5, 2) == Path(
        "/tmp/out/raw/solutions/cudaq/qubo/n_5/2"
    )
    assert solutions_raw_dir(root, "simulated_annealing", "qubo", 5, None) == Path(
        "/tmp/out/raw/solutions/simulated_annealing/qubo/n_5"
    )


def test_normalise_n_cities() -> None:
    assert normalise_n_cities(4) == [4]
    assert normalise_n_cities([4, 7]) == [4, 7]


@pytest.mark.parametrize(
    ("solver", "raw", "expected"),
    [
        ("simulated_annealing", 3, [(None, 3)]),
        ("simulated_annealing", [2, 5], [(None, 2)]),
        ("cirq", 2, [(2, 2)]),
        ("cudaq", [1, 3], [(1, 1), (3, 3)]),
    ],
)
def test_experiment_depth_iterations(
    solver: str,
    raw: int | list[int],
    expected: list[tuple[int | None, int]],
) -> None:
    assert experiment_depth_iterations(solver, raw) == expected


def test_load_instance_generation_entries_smoke() -> None:
    pairs = load_instance_generation_entries(DEFAULT_INSTANCE_GENERATION_CONFIG_PATH)
    assert pairs
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in pairs)
    assert pairs == sorted(pairs, key=lambda t: t[0])


def test_parse_after_merge_experiment_style_dict() -> None:
    base = {
        "n_instances": 100,
        "solver": "cudaq",
        "formulation": "qubo",
        "optimizer": "Powell",
        "restriction": {"lambda_0": 1000.0, "lambda_1": 1000.0, "lambda_2": 1000.0},
        "qaoa_depth": 1,
        "qaoa_max_iter": 1000,
        "qaoa_delta_t": 0.55,
        "qaoa_optimizer_tol": 1.0e-6,
        "qaoa_shots": 100000,
        "qaoa_sample_shots": 100000,
        "seed": 42,
        "max_iterations": 1000,
        "timeout_seconds": None,
        "sa_t_initial": 1000.0,
        "sa_t_final": 1.0e-6,
        "sa_alpha": 0.995,
        "noise": {"enabled": False},
    }
    exp = {"n_cities": 5, "n_instances": 2, "qaoa_depth": [1, 2]}
    merged = merge_solver_yaml_dicts(base, exp)
    merged.pop("n_cities")
    qraw = merged.pop("qaoa_depth")
    for _path_d, run_d in experiment_depth_iterations(merged["solver"], qraw):
        cfg = {**merged, "qaoa_depth": run_d}
        parsed = parse_solver_config_dict(cfg)
        assert parsed["n_instances"] == 2
        assert parsed["qaoa_depth"] == run_d


def test_run_check_solution_feasibility_all_ok(tmp_path: Path) -> None:
    from experiments.main_experiment_workflow import run_check_solution_feasibility

    leaf = tmp_path / "raw" / "solutions" / "cirq" / "tqudo" / "n_4" / "1"
    leaf.mkdir(parents=True)
    (leaf / "instance_1.json").write_text(
        json.dumps({"solver_output": {"feasible": True, "solver_name": "cirq"}}), encoding="utf-8"
    )
    assert run_check_solution_feasibility(tmp_path, "cirq") == 0


def test_run_check_solution_feasibility_detects_bad(tmp_path: Path) -> None:
    from experiments.main_experiment_workflow import run_check_solution_feasibility

    base = tmp_path / "raw" / "solutions" / "cudaq" / "qubo" / "n_3"
    base.mkdir(parents=True)
    (base / "instance_1.json").write_text(
        json.dumps({"solver_output": {"feasible": True}}), encoding="utf-8"
    )
    (base / "instance_2.json").write_text(
        json.dumps({"solver_output": {"feasible": False}}), encoding="utf-8"
    )
    (base / "instance_3.json").write_text(
        json.dumps({"solver_output": {"error": "boom"}}), encoding="utf-8"
    )
    assert run_check_solution_feasibility(tmp_path, "cudaq") == 1


def test_run_check_solution_feasibility_missing_dir(tmp_path: Path) -> None:
    from experiments.main_experiment_workflow import run_check_solution_feasibility

    assert run_check_solution_feasibility(tmp_path, "simulated_annealing") == 2
