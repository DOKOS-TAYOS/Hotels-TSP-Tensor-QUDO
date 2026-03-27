"""Shared optional-dependency checks and small helpers for ``data_analysis``."""

from __future__ import annotations


def require_pandas(*, context: str) -> None:
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"{context} requires pandas (pip install -e '.[analysis]')."
        ) from exc


def coerce_bool_scalar(x: object) -> bool:
    """Normalize manifest/boolean-like cells (True/False or string ``\"true\"``)."""
    if x is True:
        return True
    if x is False:
        return False
    return str(x).lower() == "true"
