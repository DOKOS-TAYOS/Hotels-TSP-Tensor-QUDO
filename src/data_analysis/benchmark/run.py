"""Orchestrate benchmark dashboard and boxplot figure generation from disk."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from data_analysis.benchmark.figures import (
    _plot_approx_ratio_boxplots_vs_ncities,
    _plot_approx_ratio_boxplots_vs_p,
    _plot_comparison_dashboard,
    _plot_dodged_boxplot_series_vs_ncities,
    _plot_paired_four_series_boxplots_vs_p,
)
from data_analysis.benchmark.plot_serde import (
    coerce_plot_kwargs,
    read_box_vs_p_long,
    read_dashboard_stats,
    read_paired_four_vs_p,
    read_triplet_series_long,
)


class _BenchmarkImageDirs(NamedTuple):
    dashboards: Path
    approx_ratio: Path
    steps: Path
    improvement: Path
    p_opt: Path
    histogram: Path


def _ensure_benchmark_image_dirs(images_dir: Path) -> _BenchmarkImageDirs:
    images_dir.mkdir(parents=True, exist_ok=True)
    dashboards = images_dir / "dashboards"
    approx_ratio = images_dir / "approx_ratio"
    steps = images_dir / "steps"
    improvement = images_dir / "improvement"
    p_opt = images_dir / "p_opt"
    histogram = images_dir / "histogram"
    for p in (dashboards, approx_ratio, steps, improvement, p_opt, histogram):
        p.mkdir(parents=True, exist_ok=True)
    return _BenchmarkImageDirs(dashboards, approx_ratio, steps, improvement, p_opt, histogram)


def run_benchmark_plots_from_disk(plots_data: Path, images_dir: Path) -> None:
    """Load ``plots_data`` Parquets and write benchmark PNGs under *images_dir*."""
    import matplotlib.pyplot as plt

    imgs = _ensure_benchmark_image_dirs(images_dir)
    root = plots_data.resolve()

    dashboard_specs: tuple[tuple[str, Path], ...] = (
        ("cudaq_qubo_vs_cirq_tqudo_n5", imgs.dashboards),
        ("cudaq_tvirt_vs_cirq_n5", imgs.dashboards),
        ("cudaq_tvirt_vs_cirq_n9", imgs.dashboards),
    )
    for stem, dest in dashboard_specs:
        pq = root / "dashboards" / f"{stem}.parquet"
        if not pq.is_file():
            continue
        x_labels, stats_list, meta = read_dashboard_stats(pq)
        ostop = meta.get("other_panels_stats_stop")
        fig = _plot_comparison_dashboard(
            x_labels=x_labels,
            stats_list=stats_list,
            label_left=str(meta["label_left"]),
            label_right=str(meta["label_right"]),
            x_axis_label=str(meta["x_axis_label"]),
            other_panels_stats_stop=int(ostop) if ostop is not None else None,
        )
        fig.savefig(dest / f"{stem}.png", dpi=150)
        plt.close(fig)

    box_vs_p_specs: tuple[str, Path] = (
        ("approx_ratio/rho_vs_p_n5_qubo_vqaoa_nqaoa", imgs.approx_ratio),
        ("steps/cudaq_tvirt_vs_qubo_n5_vs_p", imgs.steps),
        ("steps/cudaq_tvirt_vs_cirq_n5_n9_vs_p", imgs.steps),
        ("p_opt/n5_qubo_vqaoa_nqaoa_popt_vs_p", imgs.p_opt),
    )
    for rel, dest in box_vs_p_specs:
        pq = root / f"{rel}.parquet"
        if not pq.is_file():
            continue
        series, meta = read_box_vs_p_long(pq)
        kw = coerce_plot_kwargs(dict(meta["plot_kwargs"]))
        fig = _plot_approx_ratio_boxplots_vs_p(series, **kw)
        fig.savefig(dest / f"{Path(rel).name}.png", dpi=150)
        plt.close(fig)

    triplet_specs: tuple[str, Path] = (
        ("approx_ratio/rho_vs_n_by_p", imgs.approx_ratio),
        ("steps/cirq_tqudo_firstmin_steps_vs_n_by_p", imgs.steps),
        ("p_opt/cirq_tqudo_popt_vs_n_by_p", imgs.p_opt),
        ("improvement/cirq_tqudo_rel_energy_vs_n_by_p", imgs.improvement),
        ("p_opt/cirq_tqudo_delta_popt_vs_n_by_p", imgs.p_opt),
        ("histogram/entropy_nat_vs_n_cirq_tqudo", imgs.histogram),
        ("histogram/top5_mass_vs_n_cirq_tqudo", imgs.histogram),
        ("histogram/near_bf_h1_vs_n_cirq_tqudo", imgs.histogram),
        ("histogram/energy_auc_vs_n_cirq_tqudo", imgs.histogram),
        ("histogram/steps_to_ref_eps_vs_n_cirq_tqudo", imgs.histogram),
    )
    for rel, dest in triplet_specs:
        pq = root / f"{rel}.parquet"
        if not pq.is_file():
            continue
        triple, meta = read_triplet_series_long(pq)
        kw = coerce_plot_kwargs(dict(meta["plot_kwargs"]))
        n_tick = kw.pop("n_tick_vals")
        stem = Path(rel).name
        tplot = str(meta.get("triplet_plot", "dodged_ncities"))
        if tplot == "approx_ratio_ncities":
            fig = _plot_approx_ratio_boxplots_vs_ncities(
                triple,
                n_tick_vals=list(int(x) for x in n_tick),
                **kw,
            )
        else:
            fig = _plot_dodged_boxplot_series_vs_ncities(
                triple,
                n_tick_vals=list(int(x) for x in n_tick),
                **kw,
            )
        fig.savefig(dest / f"{stem}.png", dpi=150)
        plt.close(fig)

    paired_specs: tuple[str, Path] = (
        ("improvement/paired_n5_cq_cirq_rel_energy_vs_p", imgs.improvement),
        ("p_opt/paired_n5_cq_cirq_delta_popt_vs_p", imgs.p_opt),
    )
    for rel, dest in paired_specs:
        pq = root / f"{rel}.parquet"
        if not pq.is_file():
            continue
        x_labels, series, meta = read_paired_four_vs_p(pq)
        kw = coerce_plot_kwargs(dict(meta["plot_kwargs"]))
        fig = _plot_paired_four_series_boxplots_vs_p(
            x_labels=x_labels,
            series=series,
            **kw,
        )
        stem = Path(rel).name
        fig.savefig(dest / f"{stem}.png", dpi=150)
        plt.close(fig)
