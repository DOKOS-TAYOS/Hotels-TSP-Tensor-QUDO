"""Generate figures under ``output/images/`` from ``processed/plots_data`` tables.

Requires :func:`data_analysis.prepare_plots.run_prepare_plots` after metrics.
Writes PNGs under ``images/`` (energy_history, dashboards, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_analysis._deps import require_plot_stack
from data_analysis.benchmark.run import run_benchmark_plots_from_disk
from data_analysis.energy_plots import (
    run_energy_curve_dispersion_figure,
    run_energy_history_figures_from_disk,
)
from data_analysis.extended_plots import run_extended_analysis_figures
from utils.output_paths import build_output_layout

_LEGACY_FLAT_FIGURE_NAMES: tuple[str, ...] = (
    "energy_history_mean_cudaq_qubo_vs_tqudo_virtual_n5.png",
    "energy_history_mean_cirq_tqudo_vs_cudaq_tvirt_n5.png",
    "energy_history_mean_cirq_tqudo_by_ncities.png",
    "cudaq_qubo_vs_tqudo_virtual_by_qaoa_depth.png",
    "cudaq_tqudo_virtual_vs_cirq_tqudo_by_qaoa_depth.png",
    "cudaq_tqudo_virtual_vs_cirq_tqudo_by_qaoa_depth_n9.png",
    "mean_approx_ratio_cudaq_qubo_cudaq_tvirt_cirq_tqudo_n5_by_qaoa_depth.png",
    "mean_approx_ratio_cirq_tqudo_n5_n8_cudaq_tvirt_n9_by_ncities.png",
    "cudaq_tqudo_virtual_vs_qubo_opt_steps_per_solver_optimal_n5_by_qaoa_depth.png",
    "cudaq_tqudo_virtual_vs_cirq_tqudo_opt_steps_per_solver_optimal_n5_n9_by_qaoa_depth.png",
    "cirq_tqudo_opt_steps_per_solver_optimal_by_ncities_by_qaoa_depth.png",
    "cirq_tqudo_p_opt_vs_ncities_by_qaoa_depth.png",
    "p_opt_cirq_tqudo_vs_cudaq_tvirt_n5_by_qaoa_depth.png",
    "cirq_tqudo_energy_improvement_vs_ncities_by_qaoa_depth.png",
    "cirq_tqudo_delta_p_opt_vs_ncities_by_qaoa_depth.png",
    "paired_n5_energy_improvement_cirq_tqudo_vs_cudaq_tvirt_by_depth.png",
    "paired_n5_delta_p_opt_cirq_tqudo_vs_cudaq_tvirt_by_depth.png",
)


def _remove_legacy_flat_figures(images_root: Path) -> None:
    """Drop pre–subfolder PNGs at ``images/`` root so reruns do not leave stale duplicates."""
    for name in _LEGACY_FLAT_FIGURE_NAMES:
        p = images_root / name
        if p.is_file():
            p.unlink()
    for p in images_root.glob("energy_history_mean_*.png"):
        if p.is_file():
            p.unlink()


def _plots_data_ready(plots_data: Path) -> bool:
    paired_block = any(
        (plots_data / "dashboards" / f"{s}.parquet").is_file()
        for s in (
            "cudaq_qubo_vs_tvirt_n5",
            "cudaq_tvirt_vs_cirq_n5",
            "cudaq_tvirt_vs_cirq_n9",
        )
    )
    energy_block = any((plots_data / "energy_history").glob("*.parquet"))
    return paired_block or energy_block


def run_plots(output_root: Path) -> None:
    require_plot_stack(context="plot")

    layout = build_output_layout(output_root)
    layout.images.mkdir(parents=True, exist_ok=True)
    _remove_legacy_flat_figures(layout.images)
    pdata = layout.plots_data

    if not _plots_data_ready(pdata):
        raise FileNotFoundError(
            f"No plot input tables under {pdata} (expected e.g. dashboards/*.parquet "
            "and/or energy_history/*.parquet). Run: python -m data_analysis.prepare_plots "
            f"--output-root {layout.root}"
        )

    run_benchmark_plots_from_disk(pdata, layout.images)
    img_energy = layout.images / "energy_history"
    run_energy_history_figures_from_disk(pdata / "energy_history", img_energy)
    run_energy_curve_dispersion_figure(
        layout.processed / "energy_curves_agg.parquet",
        img_energy,
    )
    run_extended_analysis_figures(layout.processed, layout.images / "extended")

    print(
        f"Figures written under {layout.images} "
        "(energy_history/, approx_ratio/, steps/, improvement/, p_opt/, histogram/, dashboards/, extended/).",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot figures from processed/plots_data (run prepare_plots first).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    run_plots(args.output_root.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
