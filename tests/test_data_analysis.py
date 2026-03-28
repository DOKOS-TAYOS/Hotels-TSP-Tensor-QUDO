"""Tests for data_analysis path parsing and ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_analysis.instance_features import instance_features_from_json_dict
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


def test_instance_features_from_json_dict() -> None:
    inst = {
        "n_cities": 4,
        "precedences": [[0, 1], [1, 2]],
        "prices_hotels": [[1.0, 2.0], [3.0, 4.0]],
        "prices_travels": [
            [
                [0.0, 5.0],
                [7.0, 0.0],
            ],
        ],
    }
    f = instance_features_from_json_dict(inst)
    assert f["inst_n_precedences"] == 2
    assert f["inst_precedence_density"] == pytest.approx(2.0 / 9.0)
    assert f["inst_prices_travels_pos_mean"] == pytest.approx((5.0 + 7.0) / 2.0)


def test_cohort_angle_stats_mean_pairwise_cosine() -> None:
    import pandas as pd

    from data_analysis.angles_stats import cohort_angle_stats

    rows = [
        {"oa_gamma": [1.0], "oa_beta": [0.0]},
        {"oa_gamma": [1.0], "oa_beta": [0.0]},
        {"oa_gamma": [0.0], "oa_beta": [1.0]},
    ]
    st = cohort_angle_stats(pd.DataFrame(rows))
    assert st["n_runs"] == 3
    assert st["n_runs_with_angles"] == 3
    assert st["n_runs_angles_dim_consistent"] == 3
    assert st["mean_pairwise_cosine"] == pytest.approx((1.0 + 0.0 + 0.0) / 3.0)


def test_cohort_angle_stats_mixed_vector_length_uses_mode_length() -> None:
    import pandas as pd

    from data_analysis.angles_stats import cohort_angle_stats

    rows = [
        {"oa_gamma": [1.0], "oa_beta": [0.0]},
        {"oa_gamma": [1.0], "oa_beta": [0.0]},
        {"oa_gamma": [0.5, 0.5], "oa_beta": [0.25, 0.25]},
    ]
    st = cohort_angle_stats(pd.DataFrame(rows))
    assert st["n_runs_with_angles"] == 3
    assert st["n_runs_angles_dim_consistent"] == 2
    assert st["angle_vector_dim_used"] == 2.0
    assert st["mean_pairwise_cosine"] == pytest.approx(1.0)


def test_json_row_instance_features_and_optimal_angles(tmp_path: Path) -> None:
    out = tmp_path / "output"
    p = out / "raw" / "solutions" / "cirq" / "tqudo" / "n_4" / "1" / "instance_1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance": {
            "n_cities": 4,
            "precedences": [[0, 1]],
            "prices_hotels": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            "prices_travels": [
                [
                    [0.0, 10.0, 10.0],
                    [20.0, 0.0, 20.0],
                    [30.0, 30.0, 0.0],
                ],
            ],
        },
        "solver_config": {"solver": "cirq", "formulation": "tqudo", "seed": 1, "qaoa_depth": 1},
        "solver_output": {
            "feasible": True,
            "objective_value": 42.0,
            "runtime_seconds": 2.0,
            "metadata": {
                "energy_history": [10.0, 9.0],
                "real_cost": 40.0,
                "optimal_angles": {"gamma": [0.5], "beta": [0.25]},
            },
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    row = json_row(p, out)
    assert row["parse_ok"] is True
    assert row["solve_ok"] is True
    assert row["inst_n_precedences"] == 1
    assert row["oa_gamma"] == [0.5]
    assert row["oa_beta"] == [0.25]
    assert json.loads(row["oa_gamma_json"] or "[]") == [0.5]


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


def test_extended_plots_writes_pngs_from_processed(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyarrow")
    import pandas as pd

    from data_analysis.extended_plots import run_extended_analysis_figures

    proc = tmp_path / "processed"
    proc.mkdir(parents=True)
    ext = tmp_path / "images" / "extended"
    paired = pd.DataFrame(
        {
            "parse_ok": [True, True],
            "solve_ok": [True, True],
            "solver": ["cirq", "cudaq"],
            "formulation": ["tqudo", "tqudo_virtual"],
            "feasible": [True, True],
            "inst_precedence_density": [0.1, 0.2],
            "approx_ratio_real": [1.01, 1.1],
            "runtime_seconds": [12.0, 30.0],
            "configs_evaluated": [50.0, 80.0],
        }
    )
    paired.to_parquet(proc / "paired_metrics.parquet", index=False)
    run_extended_analysis_figures(proc, ext)
    assert (ext / "instance_precedence_density_vs_rho.png").is_file()
    assert (ext / "efficiency_runtime_vs_rho.png").is_file()


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
