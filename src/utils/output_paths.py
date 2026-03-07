"""Output path helpers for raw, processed, and image artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Standard output folders used by the project."""

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
