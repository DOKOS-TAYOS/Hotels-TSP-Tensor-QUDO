"""Output path helpers for raw, processed, and image artifacts.

For on-disk experiment layouts (``raw/instances``, ``raw/solutions``), see
:mod:`utils.experiment_paths`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Standard output directory layout for experiment artifacts.

    Attributes:
        root: Base output directory.
        raw: Subfolder for raw JSON or solver dumps.
        processed: Subfolder for post-processed tables or aggregates.
        plots_data: Subfolder for per-figure tables fed to ``data_analysis.plot``.
        images: Subfolder for figures.

    """

    root: Path
    raw: Path
    processed: Path
    plots_data: Path
    images: Path


def build_output_layout(root: Path) -> OutputLayout:
    """Build output layout paths without creating directories.

    Args:
        root: Base directory for output artifacts.

    Returns:
        OutputLayout with root, raw, processed, plots_data, and images subpaths.

    """
    proc = root / "processed"
    return OutputLayout(
        root=root,
        raw=root / "raw",
        processed=proc,
        plots_data=proc / "plots_data",
        images=root / "images",
    )
