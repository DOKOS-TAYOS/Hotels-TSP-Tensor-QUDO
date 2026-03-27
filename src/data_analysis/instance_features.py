"""Scalar instance descriptors from experiment JSON ``instance`` objects."""

from __future__ import annotations

import math
from typing import Any


def _flat_hotels(prices_hotels: Any) -> list[float]:
    out: list[float] = []
    if not isinstance(prices_hotels, list):
        return out
    for row in prices_hotels:
        if not isinstance(row, list):
            continue
        for v in row:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def _positive_travel_costs(prices_travels: Any) -> list[float]:
    """Collect travel costs excluding diagonal entries (i == j) per time slice."""
    out: list[float] = []
    if not isinstance(prices_travels, list):
        return out
    for slice_ in prices_travels:
        if not isinstance(slice_, list):
            continue
        for i, row in enumerate(slice_):
            if not isinstance(row, list):
                continue
            for j, v in enumerate(row):
                if i == j:
                    continue
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    continue
    return out


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def instance_features_from_json_dict(instance: dict[str, Any] | None) -> dict[str, Any]:
    """Return manifest-ready scalar keys for ``instance`` (or nulls if unusable)."""
    keys = (
        "inst_n_precedences",
        "inst_precedence_density",
        "inst_prices_hotels_mean",
        "inst_prices_hotels_std",
        "inst_prices_hotels_range",
        "inst_prices_travels_pos_mean",
        "inst_prices_travels_pos_std",
    )
    empty = dict.fromkeys(keys, None)
    if not isinstance(instance, dict):
        return empty
    try:
        n_cities = int(instance["n_cities"])
    except (KeyError, TypeError, ValueError):
        return empty
    n_available = n_cities - 1
    if n_available <= 0:
        return empty

    precedences = instance.get("precedences")
    n_prec = len(precedences) if isinstance(precedences, list) else 0
    denom = float(n_available * n_available)
    density = float(n_prec) / denom if denom else None

    hotels = _flat_hotels(instance.get("prices_hotels"))
    h_mean, h_std = _mean_std(hotels)
    h_range: float | None
    if hotels:
        h_range = float(max(hotels) - min(hotels))
    else:
        h_range = None

    travels = _positive_travel_costs(instance.get("prices_travels"))
    t_mean, t_std = _mean_std(travels)

    return {
        "inst_n_precedences": int(n_prec),
        "inst_precedence_density": density,
        "inst_prices_hotels_mean": h_mean,
        "inst_prices_hotels_std": h_std,
        "inst_prices_hotels_range": h_range,
        "inst_prices_travels_pos_mean": t_mean,
        "inst_prices_travels_pos_std": t_std,
    }
