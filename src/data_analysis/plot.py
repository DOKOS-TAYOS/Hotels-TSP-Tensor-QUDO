"""Generate figures under ``output/images/`` from processed tables.

Calls :func:`data_analysis.energy_plots.run_energy_history_figures` when
``energy_curves_agg`` exists and :func:`data_analysis.benchmark_plots.run_benchmark_plots`
when ``paired_metrics`` exists (dashboards, approximation ratio, optimal-state sample
probability, relative energy improvement, and paired CQ-vs-Cirq comparisons — see
``docs/data_analysis.md`` Phase 3). PNGs are written to subfolders under ``images/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_analysis.benchmark_plots import run_benchmark_plots
from data_analysis.energy_plots import run_energy_history_figures
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


def _require_plot_deps() -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "plot requires matplotlib and pandas (pip install -e '.[analysis]')."
        ) from exc


def run_plots(output_root: Path) -> None:
    _require_plot_deps()
    import pandas as pd

    layout = build_output_layout(output_root)
    layout.images.mkdir(parents=True, exist_ok=True)
    _remove_legacy_flat_figures(layout.images)
    proc = layout.processed

    paired_path = proc / "paired_metrics.parquet"
    curves_path = proc / "energy_curves_agg.parquet"
    if not paired_path.is_file() and not curves_path.is_file():
        raise FileNotFoundError(
            f"No paired_metrics.parquet or energy_curves_agg.parquet in {proc}. Run metrics first."
        )

    p_no_sa = pd.DataFrame()
    if paired_path.is_file():
        p = pd.read_parquet(paired_path)
        p_no_sa = p[p["solver"] != "simulated_annealing"] if "solver" in p.columns else p
        if not p_no_sa.empty:
            run_benchmark_plots(p_no_sa, output_root.resolve(), layout.images)

    if curves_path.is_file():
        c = pd.read_parquet(curves_path)
        if not c.empty and "mean" in c.columns:
            run_energy_history_figures(p_no_sa, c, layout.images)

    print(
        f"Figures written under {layout.images} "
        "(energy_history/, approx_ratio/, steps/, improvement/, p_opt/, dashboards/).",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot figures from processed metrics.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    run_plots(args.output_root.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
