"""Tests for data_analysis path parsing and ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_analysis.records import json_row, path_context
from data_analysis.scan import iter_raw_json_files


def test_path_context_disk_layout(tmp_path: Path) -> None:
    out = tmp_path / "output"
    raw = out / "raw"
    p = raw / "solutions" / "cudaq" / "qubo" / "n_5" / "2" / "instance_3.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")
    ctx = path_context(p, out)
    assert ctx["layout"] == "disk"
    assert ctx["solver"] == "cudaq"
    assert ctx["formulation"] == "qubo"
    assert ctx["n_cities"] == 5
    assert ctx["qaoa_depth"] == 2
    assert ctx["instance_key"] == 3


def test_path_context_disk_no_depth(tmp_path: Path) -> None:
    out = tmp_path / "output"
    p = out / "raw" / "solutions" / "brute_force" / "tqudo" / "n_5" / "instance_1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")
    ctx = path_context(p, out)
    assert ctx["layout"] == "disk"
    assert ctx["qaoa_depth"] is None


def test_iter_raw_json_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "solutions" / "a").mkdir(parents=True)
    (raw / "solutions" / "a" / "x.json").write_text("{}", encoding="utf-8")
    (raw / "exp_20200101_120000_inst_0_cudaq_qubo.json").write_text("{}", encoding="utf-8")
    found = sorted(iter_raw_json_files(raw))
    assert len(found) == 2


def test_json_row_minimal_disk(tmp_path: Path) -> None:
    out = tmp_path / "output"
    p = out / "raw" / "solutions" / "simulated_annealing" / "tqudo" / "n_5" / "instance_1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance": {"n_cities": 5, "precedences": [], "prices_hotels": [], "prices_travels": []},
        "solver_config": {"solver": "simulated_annealing", "formulation": "tqudo", "seed": 1},
        "solver_output": {
            "feasible": True,
            "objective_value": 42.0,
            "runtime_seconds": 1.0,
            "metadata": {"energy_history": [10.0, 9.0], "real_cost": 40.0},
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    row = json_row(p, out)
    assert row["parse_ok"] is True
    assert row["solver"] == "simulated_annealing"
    assert row["n_energy_steps"] == 2
    assert row["instance_key"] == 1


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_run_ingest_writes_manifest(tmp_path: Path, fmt: str) -> None:
    pytest.importorskip("pandas")
    from data_analysis.ingest import run_ingest

    out = tmp_path / "output"
    p = out / "raw" / "solutions" / "x" / "qubo" / "n_5" / "instance_1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "instance": {"n_cities": 5},
                "solver_config": {"solver": "x", "formulation": "qubo"},
                "solver_output": {"feasible": True, "objective_value": 1.0},
            }
        ),
        encoding="utf-8",
    )
    manifest = run_ingest(out, fmt)
    assert manifest.is_file()
