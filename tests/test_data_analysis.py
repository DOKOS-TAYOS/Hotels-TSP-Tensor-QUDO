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
    p = out / "raw" / "solutions" / "cudaq" / "cudaq" / "qubo" / "n_5" / "2" / "instance_3.json"
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
        dash / "cudaq_qubo_vs_cirq_tqudo_n5.parquet",
        [empty, empty, empty],
        x_labels=["1", "2", "3"],
        label_left="L",
        label_right="R",
        x_axis_label="p",
    )
    run_plots(out)
    png = out / "images" / "dashboards" / "cudaq_qubo_vs_cirq_tqudo_n5.png"
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


def test_merge_qubo_vs_tqudo_n5_prefers_cirq_tensor_rows() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    from data_analysis.prepare_plots import _merge_qubo_vs_tqudo_n5

    paired = pd.DataFrame(
        [
            {
                "parse_ok": True,
                "solve_ok": True,
                "solver": "cudaq",
                "formulation": "qubo",
                "n_cities": 5,
                "instance_key": 1,
                "qaoa_depth": 1,
                "feasible": True,
                "real_cost": 11.0,
                "ref_real_cost": 10.0,
            },
            {
                "parse_ok": True,
                "solve_ok": True,
                "solver": "cirq",
                "formulation": "tqudo",
                "n_cities": 5,
                "instance_key": 1,
                "qaoa_depth": 1,
                "feasible": True,
                "real_cost": 9.0,
                "ref_real_cost": 9.0,
            },
            {
                "parse_ok": True,
                "solve_ok": True,
                "solver": "cudaq",
                "formulation": "tqudo_virtual",
                "n_cities": 5,
                "instance_key": 1,
                "qaoa_depth": 1,
                "feasible": False,
                "real_cost": 15.0,
                "ref_real_cost": 9.0,
            },
        ]
    )

    merged = _merge_qubo_vs_tqudo_n5(paired)

    assert len(merged) == 1
    assert merged.iloc[0]["solver_left"] == "cudaq"
    assert merged.iloc[0]["formulation_left"] == "qubo"
    assert merged.iloc[0]["solver_right"] == "cirq"
    assert merged.iloc[0]["formulation_right"] == "tqudo"
    assert bool(merged.iloc[0]["feasible_right"]) is True
    assert float(merged.iloc[0]["real_cost_right"]) == 9.0


def test_benchmark_renderer_supports_new_rho_and_popt_stems(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyarrow")

    from data_analysis.benchmark.plot_serde import write_box_vs_p_long
    from data_analysis.benchmark.run import run_benchmark_plots_from_disk

    plots_data = tmp_path / "processed" / "plots_data"
    images = tmp_path / "images"

    write_box_vs_p_long(
        plots_data / "approx_ratio" / "rho_vs_p_n5_qubo_vqaoa_nqaoa.parquet",
        [
            (r"QUBO ($n=5$)", {1: [1.02], 2: [1.03], 3: [1.01]}),
            (r"V-QAOA ($n=5$)", {1: [1.04], 2: [1.02], 3: [1.02]}),
            (r"N-QAOA ($n=5$)", {1: [1.01], 2: [1.01], 3: [1.03]}),
        ],
        plot_kwargs={"figsize": (6.9, 4.8)},
    )
    write_box_vs_p_long(
        plots_data / "p_opt" / "n5_qubo_vqaoa_nqaoa_popt_vs_p.parquet",
        [
            (r"QUBO ($n=5$)", {1: [1.0e-3], 2: [2.0e-3], 3: [3.0e-3]}),
            (r"V-QAOA ($n=5$)", {1: [2.0e-2], 2: [4.0e-2], 3: [5.0e-2]}),
            (r"N-QAOA ($n=5$)", {1: [2.5e-2], 2: [4.2e-2], 3: [6.5e-2]}),
        ],
        plot_kwargs={
            "y_label": r"$P(\mathrm{opt})$",
            "y_axis_kind": "generic",
            "y_scale": "log",
            "log_y_clip_upper": 1.0,
            "figsize": (6.9, 4.8),
            "uniform_p_opt_hline_ns": (5,),
            "uniform_qubo_p_opt_hline_ns": (5,),
            "uniform_refs_in_ylim": True,
        },
    )

    run_benchmark_plots_from_disk(plots_data, images)

    assert (images / "approx_ratio" / "rho_vs_p_n5_qubo_vqaoa_nqaoa.png").is_file()
    assert (images / "p_opt" / "n5_qubo_vqaoa_nqaoa_popt_vs_p.png").is_file()


def test_p_opt_lists_by_depth_unpaired_discards_zero_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    from data_analysis.benchmark import collectors

    paired = pd.DataFrame(
        [
            {
                "parse_ok": True,
                "solve_ok": True,
                "solver": "cudaq",
                "formulation": "tqudo_virtual",
                "n_cities": 9,
                "instance_key": 1,
                "qaoa_depth": 1,
                "has_final_samples": True,
            },
            {
                "parse_ok": True,
                "solve_ok": True,
                "solver": "cudaq",
                "formulation": "tqudo_virtual",
                "n_cities": 9,
                "instance_key": 2,
                "qaoa_depth": 1,
                "has_final_samples": True,
            },
        ]
    )

    values = iter([0.0, 1.25e-5])

    def _fake_p_opt_final_from_row(*args: object, **kwargs: object) -> float:
        return float(next(values))

    monkeypatch.setattr(collectors, "_p_opt_final_from_row", _fake_p_opt_final_from_row)

    out = collectors._p_opt_lists_by_depth_unpaired(
        paired,
        solver="cudaq",
        formulation="tqudo_virtual",
        n_cities=9,
        output_root=Path("."),
        bf_cache={},
    )

    assert out == {1: [pytest.approx(1.25e-5)]}


def test_energy_improvement_lists_by_depth_unpaired_keeps_vqaoa_n9_p3() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    from data_analysis.benchmark.collectors import _metric_lists_by_depth_unpaired

    paired = pd.DataFrame(
        [
            {
                "parse_ok": True,
                "solve_ok": True,
                "solver": "cudaq",
                "formulation": "tqudo_virtual",
                "n_cities": 9,
                "instance_key": 1,
                "qaoa_depth": 3,
                "energy_improvement_rel": 0.41,
            },
            {
                "parse_ok": True,
                "solve_ok": True,
                "solver": "cudaq",
                "formulation": "tqudo_virtual",
                "n_cities": 9,
                "instance_key": 2,
                "qaoa_depth": 3,
                "energy_improvement_rel": 0.52,
            },
        ]
    )

    out = _metric_lists_by_depth_unpaired(
        paired,
        solver="cudaq",
        formulation="tqudo_virtual",
        n_cities=9,
        metric_col="energy_improvement_rel",
    )

    assert out == {3: [pytest.approx(0.41), pytest.approx(0.52)]}


def test_dashboard_hides_empty_optimal_panel_and_simplifies_cost_label() -> None:
    pytest.importorskip("matplotlib")

    from data_analysis.benchmark.figures import _plot_comparison_dashboard

    stats = [
        {
            "left_optimal": 0,
            "left_feasible_subopt": 4,
            "left_infeasible": 96,
            "right_optimal": 0,
            "right_feasible_subopt": 3,
            "right_infeasible": 97,
            "cost_left_better_cond": 2,
            "cost_right_better_cond": 2,
            "cost_tie_cond": 0,
            "n_both_feasible": 4,
            "only_left_feasible": 1,
            "only_right_feasible": 0,
            "n_paired": 100,
            "only_left_optimal": 0,
            "only_right_optimal": 0,
        },
        {
            "left_optimal": 0,
            "left_feasible_subopt": 8,
            "left_infeasible": 92,
            "right_optimal": 0,
            "right_feasible_subopt": 6,
            "right_infeasible": 94,
            "cost_left_better_cond": 3,
            "cost_right_better_cond": 3,
            "cost_tie_cond": 0,
            "n_both_feasible": 6,
            "only_left_feasible": 2,
            "only_right_feasible": 1,
            "n_paired": 100,
            "only_left_optimal": 0,
            "only_right_optimal": 0,
        },
        {
            "left_optimal": 0,
            "left_feasible_subopt": 5,
            "left_infeasible": 95,
            "right_optimal": 0,
            "right_feasible_subopt": 0,
            "right_infeasible": 0,
            "cost_left_better_cond": 0,
            "cost_right_better_cond": 0,
            "cost_tie_cond": 0,
            "n_both_feasible": 0,
            "only_left_feasible": 0,
            "only_right_feasible": 0,
            "n_paired": 100,
            "only_left_optimal": 0,
            "only_right_optimal": 0,
        },
    ]

    fig = _plot_comparison_dashboard(
        x_labels=["1", "2", "3"],
        stats_list=stats,
        label_left="V-QAOA",
        label_right="N-QAOA",
        x_axis_label="p",
        other_panels_stats_stop=2,
    )

    assert fig.axes[1].get_ylabel() == "Instances"
    assert fig.axes[3].axison is False


def test_values_outside_boxplot_whiskers_keeps_only_true_outliers() -> None:
    import numpy as np

    from data_analysis.benchmark.figures import _values_outside_boxplot_whiskers

    visible = _values_outside_boxplot_whiskers([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

    assert np.allclose(visible, np.array([100.0]))


def test_scatter_points_render_above_box_artists() -> None:
    from data_analysis.benchmark.figures import (
        _ZORDER_BOX_ARTISTS,
        _ZORDER_STRIP_SCATTER_POINTS,
    )

    assert _ZORDER_STRIP_SCATTER_POINTS > _ZORDER_BOX_ARTISTS
