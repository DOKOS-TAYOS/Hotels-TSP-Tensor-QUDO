"""Discover raw experiment JSON files under an output tree."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def iter_raw_json_files(raw_dir: Path) -> Iterator[Path]:
    """Yield solution JSON paths under ``raw/solutions/**/*.json``.

    Args:
        raw_dir: The ``raw`` directory inside an output root.

    Yields:
        Paths to ``*.json`` files, sorted by the caller if needed.

    """
    solutions = raw_dir / "solutions"
    if solutions.is_dir():
        yield from solutions.rglob("*.json")
