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
) -> None:
    """Run ingest → metrics → plots on *output_root*."""
    root = output_root.resolve()
    run_ingest(root, manifest_format)
    run_metrics(root, sample_quality=sample_quality)
    if not skip_plots:
        run_plots(root)


def process_raw_results(raw_dir: Path, processed_dir: Path) -> None:
    """Backward-compatible entry: treat *raw_dir* parent as output root.

    Args:
        raw_dir: Typically ``.../output/raw``.
        processed_dir: Target ``processed`` directory (must equal ``output/processed``).

    """
    if processed_dir.name != "processed":
        raise ValueError("processed_dir must be named 'processed' for layout consistency.")
    output_root = raw_dir.parent.resolve()
    run_pipeline(output_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Full data analysis pipeline.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--sample-quality", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    run_pipeline(
        args.output_root.resolve(),
        manifest_format=args.format,
        sample_quality=args.sample_quality,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
