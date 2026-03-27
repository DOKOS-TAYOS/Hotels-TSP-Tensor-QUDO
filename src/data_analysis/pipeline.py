"""Orchestrate ingest, metrics, and plotting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_analysis.ingest import run_ingest
from data_analysis.metrics import run_metrics
from data_analysis.plot import run_plots


def run_pipeline(
    output_root: Path,
    manifest_format: str = "parquet",
    sample_quality: bool = False,
    skip_plots: bool = False,
    *,
    energy_curve_percentiles: bool = True,
) -> None:
    """Run ingest → metrics → plots on *output_root*."""
    root = output_root.resolve()
    run_ingest(root, manifest_format)
    run_metrics(
        root,
        sample_quality=sample_quality,
        energy_curve_percentiles=energy_curve_percentiles,
    )
    if not skip_plots:
        run_plots(root)


def process_raw_results(raw_dir: Path, processed_dir: Path) -> None:
    """Backward-compatible entry: infer output root from *raw_dir* and validate *processed_dir*.

    Args:
        raw_dir: Typically ``.../output/raw`` (sibling of ``processed``).
        processed_dir: Must be ``<output_root>/processed`` (name ``processed``, same parent as *raw_dir*).

    """
    raw_resolved = raw_dir.resolve()
    proc_resolved = processed_dir.resolve()
    if proc_resolved.name != "processed":
        raise ValueError("processed_dir must be named 'processed' for layout consistency.")
    output_root = raw_resolved.parent
    if proc_resolved.parent != output_root:
        raise ValueError(
            "processed_dir must be the processed/ directory under the same output root as raw_dir "
            f"(expected {output_root / 'processed'}, got {proc_resolved})."
        )
    run_pipeline(output_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Full data analysis pipeline.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--sample-quality", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--no-energy-curve-percentiles",
        action="store_true",
        help="Omit p25/p50/p75 from energy_curves_agg (faster; plots use mean/std only).",
    )
    args = parser.parse_args(argv)
    run_pipeline(
        args.output_root.resolve(),
        manifest_format=args.format,
        sample_quality=args.sample_quality,
        skip_plots=args.skip_plots,
        energy_curve_percentiles=not args.no_energy_curve_percentiles,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
