"""Logging utilities for experiment scripts and notebooks."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a shared timestamped format.

    Args:
        level: Name such as ``INFO`` or ``DEBUG``; invalid names fall back to INFO.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
