"""Logging utilities for experiment scripts and notebooks."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure project-wide logging with a single consistent format."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
