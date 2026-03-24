"""Recursive JSON-friendly normalization (NaN/inf, numpy, dataclasses)."""

from __future__ import annotations

import dataclasses
from typing import Any


def to_json_friendly(obj: Any) -> Any:
    """Recursively normalise *obj* for JSON encoding.

    Non-finite floats become ``None`` (JSON has no NaN/inf).
    """
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [to_json_friendly(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_friendly(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_json_friendly(dataclasses.asdict(obj))
    return obj
