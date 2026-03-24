"""Generate figures under ``output/images/`` from processed tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_analysis.output_paths import build_output_layout


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
    import matplotlib.pyplot as plt
    import pandas as pd

    layout = build_output_layout(output_root)
    layout.images.mkdir(parents=True, exist_ok=True)
    proc = layout.processed

    summary_path = proc / "summary_by_config.csv"
    paired_path = proc / "paired_metrics.parquet"
    if not summary_path.is_file() and not paired_path.is_file():
        raise FileNotFoundError(
            f"No summary_by_config.csv or paired_metrics.parquet in {proc}. Run metrics first."
        )

    if summary_path.is_file() and summary_path.stat().st_size > 0:
        try:
            s = pd.read_csv(summary_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            s = pd.DataFrame()
        if not s.empty and "feas_rate" in s.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            labels = (
                s["solver"].astype(str)
                + " / "
                + s["formulation"].astype(str)
                + " / n="
                + s["n_cities"].astype(str)
            )
            ax.barh(range(len(s)), s["feas_rate"].fillna(0.0))
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Feasibility rate")
            ax.set_title("Feasible solutions by configuration")
            ax.set_xlim(0, 1.05)
            fig.tight_layout()
            fig.savefig(layout.images / "feasibility_by_config.png", dpi=150)
            plt.close(fig)

    if paired_path.is_file():
        p = pd.read_parquet(paired_path)
        if (
            not p.empty
            and "approx_ratio_real" in p.columns
            and "parse_ok" in p.columns
            and "solver" in p.columns
        ):
            sub = p[
                p["parse_ok"]
                & (p["solver"] != "brute_force")
                & p["approx_ratio_real"].notna()
            ]
        else:
            sub = pd.DataFrame()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.violinplot(
                [sub["approx_ratio_real"].clip(0, 3).values],
                positions=[0],
                showmeans=True,
            )
            ax.set_xticks([0])
            ax.set_xticklabels(["approx_ratio_real vs brute_force ref"])
            ax.set_ylabel("Ratio (clipped to 3 for display)")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            fig.tight_layout()
            fig.savefig(layout.images / "approx_ratio_real_violin.png", dpi=150)
            plt.close(fig)

    curves_path = proc / "energy_curves_agg.parquet"
    if curves_path.is_file():
        c = pd.read_parquet(curves_path)
        if not c.empty and "p50" in c.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            for grp, part in c.groupby(["solver", "formulation", "n_cities"], dropna=False):
                part = part.sort_values("step")
                label = f"{grp[0]} / {grp[1]} / n={grp[2]}"
                ax.plot(part["step"], part["p50"], label=label[:60])
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Median energy (scaled units)")
            ax.legend(fontsize=7, loc="best")
            ax.set_title("Energy history (median by configuration)")
            fig.tight_layout()
            fig.savefig(layout.images / "energy_history_median.png", dpi=150)
            plt.close(fig)

    print(f"Figures written to {layout.images}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot figures from processed metrics.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    run_plots(args.output_root.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
