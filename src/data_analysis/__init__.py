"""Data analysis: ingest, metrics, ``processed/plots_data``, figures under ``output/images``.

Plots (see ``docs/data_analysis.md``) read ``plots_data`` Parquet tables produced by
``data_analysis.prepare_plots`` from paired metrics and energy aggregates.
"""

from __future__ import annotations

__all__ = ["process_raw_results", "run_pipeline"]


def __getattr__(name: str) -> object:
    """Lazy-export ``process_raw_results`` and ``run_pipeline``.

    Args:
        name: Attribute requested on ``data_analysis``.

    Returns:
        The imported callable for a supported name.

    Raises:
        AttributeError: If ``name`` is not a lazy export.

    """
    if name == "process_raw_results":
        from data_analysis.pipeline import process_raw_results

        return process_raw_results
    if name == "run_pipeline":
        from data_analysis.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
