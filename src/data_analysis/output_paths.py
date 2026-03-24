"""Output layout helpers (duplicated from ``utils.output_paths`` to avoid import cycles)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Standard output directory layout for experiment artifacts."""

    root: Path
    raw: Path
    processed: Path
    images: Path


def build_output_layout(root: Path) -> OutputLayout:
    """Build output layout paths without creating directories."""
    return OutputLayout(
        root=root,
        raw=root / "raw",
        processed=root / "processed",
        images=root / "images",
    )
