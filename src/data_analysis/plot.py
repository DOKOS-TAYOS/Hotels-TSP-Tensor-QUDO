"""Generate figures under ``output/images/`` from processed tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_analysis.benchmark_plots import run_benchmark_plots
from data_analysis.energy_plots import run_energy_history_figures
from utils.output_paths import build_output_layout


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

    print(f"Figures written to {layout.images}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot figures from processed metrics.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    run_plots(args.output_root.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
