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


def test_path_context_disk_duplicate_solver_dir(tmp_path: Path) -> None:
    """Workflows may emit raw/solutions/{solver}/{solver}/formulation/..."""
    out = tmp_path / "output"
    p = (
        out
        / "raw"
        / "solutions"
        / "cudaq"
        / "cudaq"
        / "qubo"
        / "n_5"
        / "2"
        / "instance_3.json"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")
    ctx = path_context(p, out)
    assert ctx["layout"] == "disk"
    assert ctx["solver"] == "cudaq"
    assert ctx["formulation"] == "qubo"
    assert ctx["n_cities"] == 5
    assert ctx["qaoa_depth"] == 2
    assert ctx["instance_key"] == 3


def test_first_optimizer_step_reaching_min_energy() -> None:
    from data_analysis.metrics import first_optimizer_step_reaching_min_energy

    assert first_optimizer_step_reaching_min_energy([3.0, 2.0, 2.5]) == 2
    assert first_optimizer_step_reaching_min_energy([1.0]) == 1
    assert first_optimizer_step_reaching_min_energy([]) is None


def test_process_raw_results_validates_processed_dir(tmp_path: Path) -> None:
    from data_analysis.pipeline import process_raw_results

    out = tmp_path / "output"
    (out / "raw").mkdir(parents=True)
    other = tmp_path / "other_output" / "processed"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="same output root"):
        process_raw_results(out / "raw", other)

    with pytest.raises(ValueError, match="named 'processed'"):
        process_raw_results(out / "raw", out / "not_processed")


def test_iter_raw_json_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "solutions" / "a").mkdir(parents=True)
    (raw / "solutions" / "a" / "x.json").write_text("{}", encoding="utf-8")
    found = sorted(iter_raw_json_files(raw))
    assert len(found) == 1


def test_json_row_solver_output_error(tmp_path: Path) -> None:
    """Failed solve JSON: parse_ok True, solve_ok False."""
    out = tmp_path / "output"
    p = out / "raw" / "solutions" / "cudaq" / "qubo" / "n_5" / "instance_1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance": {"n_cities": 5},
        "solver_config": {"solver": "cudaq", "formulation": "qubo"},
        "solver_output": {"solver_name": "cudaq", "error": "boom"},
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    row = json_row(p, out)
    assert row["parse_ok"] is True
    assert row["solve_ok"] is False
    assert "boom" in (row.get("solver_error") or "")


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
    assert row["solve_ok"] is True
    assert row["solver"] == "simulated_annealing"
    assert row["n_energy_steps"] == 2
    assert row["instance_key"] == 1


def test_prepare_plots_requires_paired_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    from data_analysis.prepare_plots import run_prepare_plots

    out = tmp_path / "output"
    (out / "processed").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="paired_metrics"):
        run_prepare_plots(out)


def test_plot_requires_plots_data_tables(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from data_analysis.plot import run_plots

    out = tmp_path / "output"
    (out / "processed" / "plots_data").mkdir(parents=True)
    (out / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="prepare_plots"):
        run_plots(out)


def test_plot_renders_from_minimal_plots_data(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyarrow")
    import pandas as pd

    from data_analysis.benchmark.pairing import _stats_from_rows
    from data_analysis.benchmark.plot_serde import write_dashboard_stats
    from data_analysis.plot import run_plots

    out = tmp_path / "output"
    dash = out / "processed" / "plots_data" / "dashboards"
    dash.mkdir(parents=True)
    empty = _stats_from_rows(pd.DataFrame())
    write_dashboard_stats(
        dash / "cudaq_qubo_vs_tvirt_n5.parquet",
        [empty, empty, empty],
        x_labels=["1", "2", "3"],
        label_left="L",
        label_right="R",
        x_axis_label="p",
    )
    run_plots(out)
    png = out / "images" / "dashboards" / "cudaq_qubo_vs_tvirt_n5.png"
    assert png.is_file()


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
