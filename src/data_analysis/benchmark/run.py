"""Orchestrate benchmark dashboard and boxplot figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

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
    pd_notna_n,
)
from data_analysis.benchmark.figures import (
    _plot_approx_ratio_boxplots_vs_ncities,
    _plot_approx_ratio_boxplots_vs_p,
    _plot_comparison_dashboard,
    _plot_dodged_boxplot_series_vs_ncities,
    _plot_paired_four_series_boxplots_vs_p,
)
from data_analysis.benchmark.pairing import _merge_paired, _stats_from_rows


class _BenchmarkImageDirs(NamedTuple):
    dashboards: Path
    approx_ratio: Path
    steps: Path
    improvement: Path
    p_opt: Path


def _ensure_benchmark_image_dirs(images_dir: Path) -> _BenchmarkImageDirs:
    images_dir.mkdir(parents=True, exist_ok=True)
    dashboards = images_dir / "dashboards"
    approx_ratio = images_dir / "approx_ratio"
    steps = images_dir / "steps"
    improvement = images_dir / "improvement"
    p_opt = images_dir / "p_opt"
    for p in (dashboards, approx_ratio, steps, improvement, p_opt):
        p.mkdir(parents=True, exist_ok=True)
    return _BenchmarkImageDirs(dashboards, approx_ratio, steps, improvement, p_opt)


def _stats_list_for_depths(merged: Any, depths: tuple[int, ...]) -> list[dict[str, float | int]]:
    """Build :func:`_stats_from_rows` dicts for each QAOA depth *p* in *depths*."""
    empty = _stats_from_rows(merged.iloc[0:0])
    out: list[dict[str, float | int]] = []
    for d in depths:
        if merged.empty:
            out.append(empty)
        else:
            sub = merged[merged["qaoa_depth"].astype(int) == int(d)]
            out.append(_stats_from_rows(sub))
    return out


def _save_merged_comparison_dashboard(
    merged: Any,
    depths: tuple[int, ...],
    *,
    label_left: str,
    label_right: str,
    out_path: Path,
    plt: Any,
    x_axis_label: str = r"$p$",
) -> None:
    fig = _plot_comparison_dashboard(
        x_labels=[str(d) for d in depths],
        stats_list=_stats_list_for_depths(merged, depths),
        label_left=label_left,
        label_right=label_right,
        x_axis_label=x_axis_label,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_benchmark_plots(paired: Any, output_root: Path, images_dir: Path) -> None:
    """Write CUDA-Q / cross-backend dashboard and mean approximation-ratio figures."""
    import matplotlib.pyplot as plt

    imgs = _ensure_benchmark_image_dirs(images_dir)
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

    for merged, lab_l, lab_r, fname in (
        (cq_merged, "QUBO", "TQUDO qubits", "cudaq_qubo_vs_tvirt_n5.png"),
        (xc_by_n[5], "TQUDO qubits", "TQUDO qudits", "cudaq_tvirt_vs_cirq_n5.png"),
        (xc_by_n[9], "TQUDO qubits", "TQUDO qudits", "cudaq_tvirt_vs_cirq_n9.png"),
    ):
        _save_merged_comparison_dashboard(
            merged,
            depths,
            label_left=lab_l,
            label_right=lab_r,
            out_path=imgs.dashboards / fname,
            plt=plt,
        )

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
    fig = _plot_approx_ratio_boxplots_vs_p(
        [
            (r"QUBO ($n=5$)", rho_q),
            (r"TQUDO qubits ($n=5$)", rho_tv5),
            (r"TQUDO qudits ($n=5$)", rho_ci5),
            (r"TQUDO qubits ($n=9$)", rho_tv9),
            (r"TQUDO qudits ($n=9$)", rho_ci9),
        ],
        figsize=(7.8, 7.8),
    )
    fig.savefig(
        imgs.approx_ratio / "n5_qubo_tvirt_cirq_vs_p.png",
        dpi=150,
    )
    plt.close(fig)

    n_multi = [5, 6, 7, 8, 9]
    vs_n_box = _approx_ratio_box_series_vs_ncities_by_depth(paired, n_values=n_multi)
    fig = _plot_approx_ratio_boxplots_vs_ncities(
        vs_n_box,
        n_tick_vals=n_multi,
        figsize=(8.5, 5),
    )
    fig.savefig(
        imgs.approx_ratio / "rho_vs_n_by_p.png",
        dpi=150,
    )
    plt.close(fig)

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
    fig_cq = _plot_approx_ratio_boxplots_vs_p(
        [
            ("QUBO", _step_lists_to_depth_dict(depths, lr_cq)),
            ("TQUDO qubits", _step_lists_to_depth_dict(depths, ll_cq)),
        ],
        y_label="steps",
        y_axis_kind="generic",
        y_scale="linear",
        figsize=(6.9, 6.9),
    )
    fig_cq.savefig(
        imgs.steps / "cudaq_tvirt_vs_qubo_n5_vs_p.png",
        dpi=150,
    )
    plt.close(fig_cq)

    ll5, lr5 = _collect_side_opt_step_lists_by_depth(xc_by_n[5], depths=depths, output_root=root)
    ll9, lr9 = _collect_side_opt_step_lists_by_depth(xc_by_n[9], depths=depths, output_root=root)
    fig_xc = _plot_approx_ratio_boxplots_vs_p(
        [
            (r"TQUDO qubits ($n=5$)", _step_lists_to_depth_dict(depths, ll5)),
            (r"TQUDO qudits ($n=5$)", _step_lists_to_depth_dict(depths, lr5)),
            (r"TQUDO qubits ($n=9$)", _step_lists_to_depth_dict(depths, ll9)),
            (r"TQUDO qudits ($n=9$)", _step_lists_to_depth_dict(depths, lr9)),
        ],
        y_label="steps",
        y_axis_kind="generic",
        y_scale="linear",
        figsize=(6.9, 6.9),
    )
    fig_xc.savefig(
        imgs.steps / "cudaq_tvirt_vs_cirq_n5_n9_vs_p.png",
        dpi=150,
    )
    plt.close(fig_xc)

    opt_steps_nc_box = _collect_cirq_tqudo_opt_steps_box_series_vs_ncities(
        paired,
        n_values=[5, 6, 7, 8, 9],
        depth_values=depths,
        output_root=root,
    )
    if opt_steps_nc_box:
        fig_ci = _plot_dodged_boxplot_series_vs_ncities(
            opt_steps_nc_box,
            n_tick_vals=[5, 6, 7, 8, 9],
            y_label="steps",
            figsize=(8.5, 5.0),
        )
        fig_ci.savefig(
            imgs.steps / "cirq_tqudo_firstmin_steps_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig_ci)

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
        fig = _plot_dodged_boxplot_series_vs_ncities(
            series_popt_n_box,
            n_tick_vals=n_tick_cirq_tqudo,
            y_label=r"$P(\mathrm{opt})$",
            figsize=(8.5, 5),
            y_scale="log",
            log_y_clip_upper=1.0,
            uniform_p_opt_vline_ns=list(n_tick_cirq_tqudo),
        )
        fig.savefig(
            imgs.p_opt / "cirq_tqudo_popt_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig)

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
    popt_n5_series = [
        ("QUBO", popt_q5),
        ("TQUDO qubits", popt_cq5),
        ("TQUDO qudits", popt_ci5),
    ]
    _popt_n5_common_kw: dict[str, Any] = {
        "y_label": r"$P(\mathrm{opt})$",
        "y_axis_kind": "generic",
        "y_scale": "log",
        "log_y_clip_upper": 1.0,
        "figsize": (6.9, 6.9),
        "uniform_p_opt_hline_ns": (5, 9),
        "uniform_qubo_p_opt_hline_ns": (5,),
    }
    for uniform_refs_in_ylim, rel_name in (
        (True, "n5_cirq_vs_cq_tvirt_popt_vs_p.png"),
        (False, "n5_cirq_vs_cq_tvirt_popt_vs_p_ydata.png"),
    ):
        fig_p = _plot_approx_ratio_boxplots_vs_p(
            popt_n5_series,
            uniform_refs_in_ylim=uniform_refs_in_ylim,
            **_popt_n5_common_kw,
        )
        fig_p.savefig(imgs.p_opt / rel_name, dpi=150)
        plt.close(fig_p)

    series_eimp_n_box = _collect_energy_improvement_box_series_vs_ncities(
        paired,
        solver="cirq",
        formulation="tqudo",
        depth_values=depths,
    )
    if series_eimp_n_box and n_tick_cirq_tqudo:
        fig = _plot_dodged_boxplot_series_vs_ncities(
            series_eimp_n_box,
            n_tick_vals=n_tick_cirq_tqudo,
            y_label=r"$(E_0 - E^\star) / |E_0|$",
            figsize=(8.5, 5),
        )
        fig.savefig(
            imgs.improvement / "cirq_tqudo_rel_energy_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig)

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
        fig = _plot_dodged_boxplot_series_vs_ncities(
            series_dp_n_box,
            n_tick_vals=n_tick_cirq_tqudo,
            y_label=r"$\Delta P(\mathrm{opt})$",
            figsize=(8.5, 5),
            y_scale="asinh",
            symlog_linthresh=2e-3,
            manual_y_limits=False,
        )
        fig.savefig(
            imgs.p_opt / "cirq_tqudo_delta_popt_vs_n_by_p.png",
            dpi=150,
        )
        plt.close(fig)

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
    fig = _plot_paired_four_series_boxplots_vs_p(
        x_labels=x_labels,
        series=[
            (r"TQUDO qubits ($n=5$)", eimp_l5),
            (r"TQUDO qudits ($n=5$)", eimp_r5),
            (r"TQUDO qubits ($n=9$)", eimp_l9),
            (r"TQUDO qudits ($n=9$)", eimp_r9),
        ],
        y_label=r"$(E_0 - E^\star) / |E_0|$",
        x_axis_label=r"$p$",
        figsize=(6.9, 6.9),
    )
    fig.savefig(
        imgs.improvement / "paired_n5_cq_cirq_rel_energy_vs_p.png",
        dpi=150,
    )
    plt.close(fig)

    dpl5_lists, dpr5_lists = _paired_delta_p_opt_lists_by_depth(
        xc_by_n[5],
        depths=depths,
        output_root=root,
        bf_cache=bf_cache,
    )
    fig = _plot_paired_four_series_boxplots_vs_p(
        x_labels=x_labels,
        series=[
            ("TQUDO qubits", dpl5_lists),
            ("TQUDO qudits", dpr5_lists),
        ],
        y_label=r"$\Delta P(\mathrm{opt})$",
        x_axis_label=r"$p$",
        figsize=(8.5, 4.9),
        y_scale="symlog",
        symlog_linthresh=1e-4,
    )
    fig.savefig(
        imgs.p_opt / "paired_n5_cq_cirq_delta_popt_vs_p.png",
        dpi=150,
    )
    plt.close(fig)
