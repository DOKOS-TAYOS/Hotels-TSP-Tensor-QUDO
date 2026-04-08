"""Shared optional-dependency checks and small helpers for ``data_analysis``."""

from __future__ import annotations


def require_pandas(*, context: str) -> None:
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{context} requires pandas (pip install -e '.[analysis]').") from exc


def require_matplotlib(*, context: str) -> None:
    try:
        import matplotlib.pyplot as _plt  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{context} requires matplotlib (pip install -e '.[analysis]').") from exc


def require_plot_stack(*, context: str) -> None:
    """``pandas`` + ``matplotlib`` (figures from processed metrics)."""
    require_pandas(context=context)
    require_matplotlib(context=context)


def coerce_bool_scalar(x: object) -> bool:
    """Normalize manifest/boolean-like cells (True/False or string ``\"true\"``)."""
    if x is True:
        return True
    if x is False:
        return False
    return str(x).lower() == "true"
