"""Data analysis: ingest raw JSON, metrics, plots under ``output/processed`` and ``output/images``.

Plots (see ``docs/data_analysis.md``) include energy-curve aggregates, CUDA-Q/Cirq
dashboards, approximation ratios, and figures for ground-state sample probability
and energy / ΔP improvements derived from ``initial_samples`` /
``final_samples`` and brute-force TQUDO references.
"""

from __future__ import annotations

__all__ = ["process_raw_results", "run_pipeline"]


def __getattr__(name: str) -> object:
    if name == "process_raw_results":
        from data_analysis.pipeline import process_raw_results

        return process_raw_results
    if name == "run_pipeline":
        from data_analysis.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
