"""Build ``processed/plots_data`` tables from paired metrics and energy aggregates."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from data_analysis._deps import require_pandas
from data_analysis.benchmark.collectors import (
    _approx_ratio_box_series_vs_ncities_by_depth,
    _approx_ratio_lists_by_depth_unpaired,
    _collect_cirq_tqudo_opt_steps_box_series_vs_ncities,
    _collect_energy_improvement_box_series_vs_ncities,
    _collect_numeric_box_series_vs_ncities,
    _collect_side_opt_step_lists_by_depth,
    _delta_p_opt_from_row,
    _paired_delta_p_opt_lists_by_depth,
    _paired_metric_lists_by_depth,
    _p_opt_final_from_row,
    _p_opt_lists_by_depth_unpaired,
    _step_lists_to_depth_dict,
    float_metric_from_paired_column,
    pd_notna_n,
)
from data_analysis.benchmark.plot_serde import (
    write_box_vs_p_long,
    write_dashboard_stats,
    write_paired_four_vs_p,
    write_triplet_series_long,
)
from data_analysis.benchmark.pairing import _merge_paired, _stats_from_rows
from data_analysis.energy_plots import write_energy_history_plot_tables
from utils.output_paths import build_output_layout


def _stats_list_for_depths(merged: Any, depths: tuple[int, ...]) -> list[dict[str, float | int]]:
    empty = _stats_from_rows(merged.iloc[0:0])
    out: list[dict[str, float | int]] = []
    for d in depths:
        if merged.empty:
            out.append(empty)
        else:
            sub = merged[merged["qaoa_depth"].astype(int) == int(d)]
            out.append(_stats_from_rows(sub))
    return out


def _ensure_plot_data_subdirs(plots_data: Path) -> None:
    for name in (
        "dashboards",
        "approx_ratio",
        "steps",
        "improvement",
        "p_opt",
        "energy_history",
        "histogram",
    ):
        (plots_data / name).mkdir(parents=True, exist_ok=True)


def write_histogram_and_trajectory_plot_tables(
    paired: Any, output_root: Path, plots_data: Path
) -> None:
    """Boxplot inputs for sample-distribution and energy-trajectory columns (if present)."""
    root = output_root.resolve()
    bf_cache: dict[tuple[int, int], list[int] | None] = {}
    depths = (1, 2, 3)
    hist_dir = plots_data / "histogram"
    hist_dir.mkdir(parents=True, exist_ok=True)

    m_ct = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == "cirq")
        & (paired["formulation"] == "tqudo")
        & paired["qaoa_depth"].notna()
    )
    n_tick = sorted({int(x) for x in paired.loc[m_ct, "n_cities"].unique() if pd_notna_n(x)})
    if not n_tick:
        n_tick = [5, 6, 7, 8, 9]

    def _triplet_if_col(col: str, stem: str, y_label: str) -> None:
        if col not in paired.columns:
            return
        series = _collect_numeric_box_series_vs_ncities(
            paired,
            solver="cirq",
            formulation="tqudo",
            depth_values=depths,
            output_root=root,
            value_fn=float_metric_from_paired_column(col),
            bf_cache=bf_cache,
        )
        if not series:
            return
        write_triplet_series_long(
            hist_dir / f"{stem}.parquet",
            series,
            plot_kwargs={"n_tick_vals": n_tick, "y_label": y_label, "figsize": (8.5, 5)},
            kind="triplet_vs_x",
        )

    _triplet_if_col("final_sample_entropy_nat", "entropy_nat_vs_n_cirq_tqudo", r"$H(\hat p)$ (nat)")
    _triplet_if_col("final_sample_top_5_mass", "top5_mass_vs_n_cirq_tqudo", "top-5 mass")
    _triplet_if_col(
        "final_sample_near_bf_mass_h1",
        "near_bf_h1_vs_n_cirq_tqudo",
        r"$P(d \leq 1)$ vs BF",
    )
    _triplet_if_col("energy_history_auc_norm", "energy_auc_vs_n_cirq_tqudo", "AUC (norm.)")
    _triplet_if_col(
        "energy_history_steps_to_ref_eps",
        "steps_to_ref_eps_vs_n_cirq_tqudo",
        r"steps to $\epsilon$-BF",
    )


def write_benchmark_plot_tables(paired: Any, output_root: Path, plots_data: Path) -> None:
    """Materialize benchmark figure inputs under *plots_data* (mirrors ``images/`` layout)."""
    root = output_root.resolve()
    bf_cache: dict[tuple[int, int], list[int] | None] = {}
    depths = (1, 2, 3)
    x_labels = [str(d) for d in depths]

    cq_merged = _merge_paired(
        paired,
        left=("cudaq", "qubo"),
        right=("cudaq", "tqudo_virtual"),
        dedupe_keys=["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
        merge_on=["n_cities", "instance_key", "qaoa_depth"],
        n_cities_filter=5,
    )
    xc_by_n = {
        n: _merge_paired(
            paired,
            left=("cudaq", "tqudo_virtual"),
            right=("cirq", "tqudo"),
            dedupe_keys=["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
            merge_on=["n_cities", "instance_key", "qaoa_depth"],
            n_cities_filter=n,
        )
        for n in (5, 9)
    }

    db = plots_data / "dashboards"
    for merged, lab_l, lab_r, stem in (
        (cq_merged, "QUBO", "TQUDO qubits", "cudaq_qubo_vs_tvirt_n5"),
        (xc_by_n[5], "TQUDO qubits", "TQUDO qudits", "cudaq_tvirt_vs_cirq_n5"),
        (xc_by_n[9], "TQUDO qubits", "TQUDO qudits", "cudaq_tvirt_vs_cirq_n9"),
    ):
        write_dashboard_stats(
            db / f"{stem}.parquet",
            _stats_list_for_depths(merged, depths),
            x_labels=x_labels,
            label_left=lab_l,
            label_right=lab_r,
            x_axis_label=r"$p$",
        )

    ar_dir = plots_data / "approx_ratio"
    rho_q = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cudaq", formulation="qubo", n_cities=5
    )
    rho_tv5 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cudaq", formulation="tqudo_virtual", n_cities=5
    )
    rho_ci5 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cirq", formulation="tqudo", n_cities=5
    )
    rho_tv9 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cudaq", formulation="tqudo_virtual", n_cities=9
    )
    rho_ci9 = _approx_ratio_lists_by_depth_unpaired(
        paired, solver="cirq", formulation="tqudo", n_cities=9
    )
    write_box_vs_p_long(
        ar_dir / "n5_qubo_tvirt_cirq_vs_p.parquet",
        [
            (r"QUBO ($n=5$)", rho_q),
            (r"TQUDO qubits ($n=5$)", rho_tv5),
            (r"TQUDO qudits ($n=5$)", rho_ci5),
            (r"TQUDO qubits ($n=9$)", rho_tv9),
            (r"TQUDO qudits ($n=9$)", rho_ci9),
        ],
        plot_kwargs={"figsize": (7.8, 7.8)},
    )

    n_multi = [5, 6, 7, 8, 9]
    vs_n_box = _approx_ratio_box_series_vs_ncities_by_depth(paired, n_values=n_multi)
    write_triplet_series_long(
        ar_dir / "rho_vs_n_by_p.parquet",
        vs_n_box,
        plot_kwargs={"n_tick_vals": n_multi, "figsize": (8.5, 5)},
        kind="triplet_vs_x",
        triplet_plot="approx_ratio_ncities",
    )

    st_dir = plots_data / "steps"
    tvirt_qubo_merged = _merge_paired(
        paired,
        left=("cudaq", "tqudo_virtual"),
        right=("cudaq", "qubo"),
        dedupe_keys=["n_cities", "instance_key", "qaoa_depth", "solver", "formulation"],
        merge_on=["n_cities", "instance_key", "qaoa_depth"],
        n_cities_filter=5,
    )
    ll_cq, lr_cq = _collect_side_opt_step_lists_by_depth(
        tvirt_qubo_merged, depths=depths, output_root=root
    )
    write_box_vs_p_long(
        st_dir / "cudaq_tvirt_vs_qubo_n5_vs_p.parquet",
        [
            ("QUBO", _step_lists_to_depth_dict(depths, lr_cq)),
            ("TQUDO qubits", _step_lists_to_depth_dict(depths, ll_cq)),
        ],
        plot_kwargs={
            "y_label": "steps",
            "y_axis_kind": "generic",
            "y_scale": "linear",
            "figsize": (6.9, 6.9),
        },
    )

    ll5, lr5 = _collect_side_opt_step_lists_by_depth(xc_by_n[5], depths=depths, output_root=root)
    ll9, lr9 = _collect_side_opt_step_lists_by_depth(xc_by_n[9], depths=depths, output_root=root)
    write_box_vs_p_long(
        st_dir / "cudaq_tvirt_vs_cirq_n5_n9_vs_p.parquet",
        [
            (r"TQUDO qubits ($n=5$)", _step_lists_to_depth_dict(depths, ll5)),
            (r"TQUDO qudits ($n=5$)", _step_lists_to_depth_dict(depths, lr5)),
            (r"TQUDO qubits ($n=9$)", _step_lists_to_depth_dict(depths, ll9)),
            (r"TQUDO qudits ($n=9$)", _step_lists_to_depth_dict(depths, lr9)),
        ],
        plot_kwargs={
            "y_label": "steps",
            "y_axis_kind": "generic",
            "y_scale": "linear",
            "figsize": (6.9, 6.9),
        },
    )

    opt_steps_nc_box = _collect_cirq_tqudo_opt_steps_box_series_vs_ncities(
        paired,
        n_values=[5, 6, 7, 8, 9],
        depth_values=depths,
        output_root=root,
    )
    if opt_steps_nc_box:
        write_triplet_series_long(
            st_dir / "cirq_tqudo_firstmin_steps_vs_n_by_p.parquet",
            opt_steps_nc_box,
            plot_kwargs={
                "n_tick_vals": [5, 6, 7, 8, 9],
                "y_label": "steps",
                "figsize": (8.5, 5.0),
            },
            kind="triplet_vs_x",
        )

    m_ct_n = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & (paired["solver"] == "cirq")
        & (paired["formulation"] == "tqudo")
        & paired["qaoa_depth"].notna()
    )
    n_tick_cirq_tqudo = sorted(
        {int(x) for x in paired.loc[m_ct_n, "n_cities"].unique() if pd_notna_n(x)}
    )
    po_dir = plots_data / "p_opt"
    im_dir = plots_data / "improvement"

    series_popt_n_box = _collect_numeric_box_series_vs_ncities(
        paired,
        solver="cirq",
        formulation="tqudo",
        depth_values=depths,
        output_root=root,
        value_fn=_p_opt_final_from_row,
        bf_cache=bf_cache,
    )
    if series_popt_n_box and n_tick_cirq_tqudo:
        write_triplet_series_long(
            po_dir / "cirq_tqudo_popt_vs_n_by_p.parquet",
            series_popt_n_box,
            plot_kwargs={
                "n_tick_vals": n_tick_cirq_tqudo,
                "y_label": r"$P(\mathrm{opt})$",
                "figsize": (8.5, 5),
                "y_scale": "log",
                "log_y_clip_upper": 1.0,
                "uniform_p_opt_vline_ns": list(n_tick_cirq_tqudo),
            },
            kind="triplet_vs_x",
        )

    popt_ci5 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cirq",
        formulation="tqudo",
        n_cities=5,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_cq5 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cudaq",
        formulation="tqudo_virtual",
        n_cities=5,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_q5 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cudaq",
        formulation="qubo",
        n_cities=5,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_ci9 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cirq",
        formulation="tqudo",
        n_cities=9,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_cq9 = _p_opt_lists_by_depth_unpaired(
        paired,
        solver="cudaq",
        formulation="tqudo_virtual",
        n_cities=9,
        output_root=root,
        bf_cache=bf_cache,
    )
    popt_box_series = [
        (r"QUBO ($n=5$)", popt_q5),
        (r"TQUDO qubits ($n=5$)", popt_cq5),
        (r"TQUDO qudits ($n=5$)", popt_ci5),
        (r"TQUDO qubits ($n=9$)", popt_cq9),
        (r"TQUDO qudits ($n=9$)", popt_ci9),
    ]
    _popt_n5_common_kw: dict[str, Any] = {
        "y_label": r"$P(\mathrm{opt})$",
        "y_axis_kind": "generic",
        "y_scale": "log",
        "log_y_clip_upper": 1.0,
        "figsize": (7.8, 7.8),
        "uniform_p_opt_hline_ns": (5, 9),
        "uniform_qubo_p_opt_hline_ns": (5,),
    }
    for uniform_refs_in_ylim, stem in (
        (True, "n5_cirq_vs_cq_tvirt_popt_vs_p"),
        (False, "n5_cirq_vs_cq_tvirt_popt_vs_p_ydata"),
    ):
        kw = {**_popt_n5_common_kw, "uniform_refs_in_ylim": uniform_refs_in_ylim}
        write_box_vs_p_long(po_dir / f"{stem}.parquet", popt_box_series, plot_kwargs=kw)

    series_eimp_n_box = _collect_energy_improvement_box_series_vs_ncities(
        paired,
        solver="cirq",
        formulation="tqudo",
        depth_values=depths,
    )
    if series_eimp_n_box and n_tick_cirq_tqudo:
        write_triplet_series_long(
            im_dir / "cirq_tqudo_rel_energy_vs_n_by_p.parquet",
            series_eimp_n_box,
            plot_kwargs={
                "n_tick_vals": n_tick_cirq_tqudo,
                "y_label": r"$(E_0 - E^\star) / |E_0|$",
                "figsize": (8.5, 5),
            },
            kind="triplet_vs_x",
        )

    series_dp_n_box = _collect_numeric_box_series_vs_ncities(
        paired,
        solver="cirq",
        formulation="tqudo",
        depth_values=depths,
        output_root=root,
        value_fn=_delta_p_opt_from_row,
        bf_cache=bf_cache,
    )
    if series_dp_n_box and n_tick_cirq_tqudo:
        write_triplet_series_long(
            po_dir / "cirq_tqudo_delta_popt_vs_n_by_p.parquet",
            series_dp_n_box,
            plot_kwargs={
                "n_tick_vals": n_tick_cirq_tqudo,
                "y_label": r"$\Delta P(\mathrm{opt})$",
                "figsize": (8.5, 5),
                "y_scale": "asinh",
                "symlog_linthresh": 2e-3,
                "manual_y_limits": False,
            },
            kind="triplet_vs_x",
        )

    eimp_l5, eimp_r5 = _paired_metric_lists_by_depth(
        xc_by_n[5],
        depths=depths,
        col_left="energy_improvement_rel_left",
        col_right="energy_improvement_rel_right",
    )
    eimp_l9, eimp_r9 = _paired_metric_lists_by_depth(
        xc_by_n[9],
        depths=depths,
        col_left="energy_improvement_rel_left",
        col_right="energy_improvement_rel_right",
    )
    write_paired_four_vs_p(
        im_dir / "paired_n5_cq_cirq_rel_energy_vs_p.parquet",
        x_labels=x_labels,
        series=[
            (r"TQUDO qubits ($n=5$)", eimp_l5),
            (r"TQUDO qudits ($n=5$)", eimp_r5),
            (r"TQUDO qubits ($n=9$)", eimp_l9),
            (r"TQUDO qudits ($n=9$)", eimp_r9),
        ],
        plot_kwargs={
            "y_label": r"$(E_0 - E^\star) / |E_0|$",
            "x_axis_label": r"$p$",
            "figsize": (6.9, 6.9),
        },
    )

    dpl5_lists, dpr5_lists = _paired_delta_p_opt_lists_by_depth(
        xc_by_n[5],
        depths=depths,
        output_root=root,
        bf_cache=bf_cache,
    )
    write_paired_four_vs_p(
        po_dir / "paired_n5_cq_cirq_delta_popt_vs_p.parquet",
        x_labels=x_labels,
        series=[
            ("TQUDO qubits", dpl5_lists),
            ("TQUDO qudits", dpr5_lists),
        ],
        plot_kwargs={
            "y_label": r"$\Delta P(\mathrm{opt})$",
            "x_axis_label": r"$p$",
            "figsize": (8.5, 4.9),
            "y_scale": "symlog",
            "symlog_linthresh": 1e-4,
        },
    )


def run_prepare_plots(output_root: Path, *, clean: bool = False) -> Path:
    """Build per-figure Parquet inputs under ``processed/plots_data``.

    Args:
        output_root: Experiment root; requires ``processed/paired_metrics.parquet``.
        clean: If True, delete existing ``plots_data`` before writing.

    Returns:
        Path to the ``plots_data`` directory.

    Raises:
        FileNotFoundError: If paired metrics are missing.

    """
    require_pandas(context="prepare_plots")
    import pandas as pd

    layout = build_output_layout(output_root.resolve())
    proc = layout.processed
    pq_p = proc / "paired_metrics.parquet"
    if not pq_p.is_file():
        raise FileNotFoundError(
            f"Missing {pq_p}. Run ingest and metrics before prepare_plots."
        )
    paired = pd.read_parquet(pq_p)
    paired_no_sa = (
        paired[paired["solver"] != "simulated_annealing"]
        if "solver" in paired.columns
        else paired
    )

    plots_data = layout.plots_data
    if clean and plots_data.is_dir():
        shutil.rmtree(plots_data)
    _ensure_plot_data_subdirs(plots_data)

    if not paired_no_sa.empty:
        write_benchmark_plot_tables(paired_no_sa, layout.root, plots_data)
        write_histogram_and_trajectory_plot_tables(paired_no_sa, layout.root, plots_data)

    curves_path = proc / "energy_curves_agg.parquet"
    curves = pd.DataFrame()
    if curves_path.is_file():
        curves = pd.read_parquet(curves_path)
    if not curves.empty and "mean" in curves.columns:
        write_energy_history_plot_tables(
            paired_no_sa,
            curves,
            plots_data / "energy_history",
        )

    print(f"Wrote plot input tables under {plots_data}", flush=True)
    return plots_data


def main(argv: list[str] | None = None) -> None:
    """CLI for the prepare_plots stage."""
    parser = argparse.ArgumentParser(
        description="Build per-figure Parquet tables under processed/plots_data from metrics.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing plots_data before writing.",
    )
    args = parser.parse_args(argv)
    run_prepare_plots(args.output_root.resolve(), clean=args.clean)


if __name__ == "__main__":
    main(sys.argv[1:])
