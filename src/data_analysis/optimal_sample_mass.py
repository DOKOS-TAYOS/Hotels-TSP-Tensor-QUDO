"""Brute-force optimal tour keys and histogram mass for P(opt) analysis plots.

Used by :mod:`data_analysis.benchmark.collectors` together with ``initial_samples`` /
``final_samples`` in solution JSON. Documented under Phase 3, section 3, in
``docs/data_analysis.md``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from utils.costs_batch import qudit_sequence_to_bitstring


def load_bruteforce_optimal_sequence(
    output_root: Path,
    n_cities: int,
    instance_key: int,
    *,
    cache: dict[tuple[int, int], list[int] | None] | None = None,
) -> list[int] | None:
    """Read optimal qudit sequence from brute_force TQUDO solution JSON."""
    ckey = (n_cities, instance_key)
    if cache is not None and ckey in cache:
        return cache[ckey]

    path = (
        output_root
        / "raw"
        / "solutions"
        / "brute_force"
        / "tqudo"
        / f"n_{n_cities}"
        / f"instance_{instance_key}.json"
    )
    if not path.is_file():
        result: list[int] | None = None
        if cache is not None:
            cache[ckey] = result
        return result

    try:
        with open(path, encoding="utf-8") as f:
            data: Any = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        result = None
        if cache is not None:
            cache[ckey] = result
        return result

    so = data.get("solver_output")
    if not isinstance(so, dict) or "error" in so:
        result = None
        if cache is not None:
            cache[ckey] = result
        return result

    meta = so.get("metadata")
    if not isinstance(meta, dict):
        result = None
        if cache is not None:
            cache[ckey] = result
        return result

    seq = meta.get("best_feasible_sequence")
    if not isinstance(seq, list):
        seq = meta.get("best_sequence")
    if not isinstance(seq, list):
        result = None
        if cache is not None:
            cache[ckey] = result
        return result

    out = []
    for x in seq:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            result = None
            if cache is not None:
                cache[ckey] = result
            return result

    if cache is not None:
        cache[ckey] = out
    return out


def native_histogram_key(sequence: list[int]) -> str:
    """TQUDO qudits (native) sample key: dash-separated qudit values."""
    return "-".join(str(int(v)) for v in sequence)


def virtual_histogram_key(sequence: list[int], n_cities: int) -> str:
    """TQUDO qubits (emulation) sample histogram key: contiguous 0/1 string."""
    d = n_cities - 1
    qubits_per_qudit = max(1, int(math.ceil(math.log2(float(d)))))
    return qudit_sequence_to_bitstring(sequence, qubits_per_qudit)


def qubo_histogram_key(sequence: list[int], n_cities: int) -> str:
    """QUBO one-hot sample key: contiguous ``'0'``/``'1'`` string of length ``(n-1)^2``.

    Matches :func:`utils.constraints.sequence_to_qubo_binary` flat index order
    ``idx = t * n_available + city`` (no import: avoids circular dependency on
    :mod:`utils.constraints` via :mod:`instance_gen_process`).
    """
    n_available = int(n_cities) - 1
    if n_available < 1:
        return ""
    bits = [0] * (n_available * n_available)
    for t in range(n_available):
        city = int(sequence[t])
        bits[t * n_available + city] = 1
    return "".join(str(b) for b in bits)


def histogram_key_for_formulation(
    sequence: list[int],
    formulation: str,
    n_cities: int,
) -> str:
    if formulation == "tqudo":
        return native_histogram_key(sequence)
    if formulation == "tqudo_virtual":
        return virtual_histogram_key(sequence, n_cities)
    if formulation == "qubo":
        return qubo_histogram_key(sequence, n_cities)
    raise ValueError(f"Unsupported formulation for histogram key: {formulation!r}")


def histogram_mass(hist: dict[str, int] | None, key: str) -> float | None:
    """Fraction of counts for ``key``, or ``None`` if histogram missing or empty."""
    if not hist or not isinstance(hist, dict):
        return None
    total = 0
    for v in hist.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total += int(v)
    if total <= 0:
        return None
    raw = hist.get(key, 0)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        cnt = 0
    else:
        cnt = int(raw)
    return float(cnt) / float(total)


def _histogram_total_counts(hist: dict[str, int] | None) -> int:
    if not hist or not isinstance(hist, dict):
        return 0
    total = 0
    for v in hist.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total += int(v)
    return total


def histogram_shannon_entropy(hist: dict[str, int] | None, base: float = math.e) -> float | None:
    """Empirical Shannon entropy :math:`-\\sum_i p_i \\log(p_i)` in nats if ``base`` is ``e``.

    Returns ``None`` if histogram is missing or has zero total counts.
    """
    if not hist or not isinstance(hist, dict):
        return None
    total = _histogram_total_counts(hist)
    if total <= 0:
        return None
    ent = 0.0
    logf = math.log
    for v in hist.values():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        c = int(v)
        if c <= 0:
            continue
        p = float(c) / float(total)
        ent -= p * logf(p) / logf(base)
    return float(ent)


def histogram_top_k_mass(hist: dict[str, int] | None, k: int) -> float | None:
    """Sum of the ``k`` largest outcome probabilities (or all if fewer than ``k`` keys)."""
    if not hist or not isinstance(hist, dict) or k <= 0:
        return None
    total = _histogram_total_counts(hist)
    if total <= 0:
        return None
    counts: list[int] = []
    for v in hist.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            counts.append(int(v))
    if not counts:
        return None
    counts.sort(reverse=True)
    take = counts[: min(k, len(counts))]
    return float(sum(take)) / float(total)


def histogram_key_hamming_distance(
    key_a: str,
    key_b: str,
    formulation: str,
    n_cities: int,
) -> int | None:
    """Assignment / bit Hamming distance between two histogram keys.

    For ``qubo`` and ``tqudo_virtual``: distance on equal-length bitstrings; ``None`` if lengths differ.
    For ``tqudo``: dash-separated qudit string vs ``native_histogram_key`` (``n_cities - 1`` parts).
    """
    n_available = int(n_cities) - 1
    if n_available < 1:
        return None
    if formulation in ("qubo", "tqudo_virtual"):
        if len(key_a) != len(key_b):
            return None
        return sum(1 for ca, cb in zip(key_a, key_b, strict=True) if ca != cb)
    if formulation == "tqudo":
        parts_a = key_a.split("-")
        parts_b = key_b.split("-")
        if len(parts_a) != n_available or len(parts_b) != n_available:
            return None
        try:
            da = [int(x) for x in parts_a]
            db = [int(x) for x in parts_b]
        except ValueError:
            return None
        return sum(1 for x, y in zip(da, db, strict=True) if x != y)
    raise ValueError(f"Unsupported formulation for key distance: {formulation!r}")


def histogram_mass_near_center(
    hist: dict[str, int] | None,
    center_key: str,
    formulation: str,
    n_cities: int,
    max_hamming: int,
) -> float | None:
    """Fraction of counts on keys within ``max_hamming`` of ``center_key`` (incompatible keys skipped)."""
    if not hist or not isinstance(hist, dict) or not center_key:
        return None
    total = _histogram_total_counts(hist)
    if total <= 0:
        return None
    near = 0
    for key, raw in hist.items():
        if not isinstance(key, str):
            continue
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        cnt = int(raw)
        if cnt <= 0:
            continue
        d = histogram_key_hamming_distance(center_key, key, formulation, n_cities)
        if d is not None and d <= max_hamming:
            near += cnt
    return float(near) / float(total)


def read_sample_histograms_from_solution_json(
    json_path: Path,
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    """Return ``(initial_samples, final_samples)`` from a solution JSON file."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data: Any = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    so = data.get("solver_output")
    if not isinstance(so, dict):
        return None, None
    meta = so.get("metadata")
    if not isinstance(meta, dict):
        return None, None
    init = meta.get("initial_samples")
    fin = meta.get("final_samples")
    init_d = init if isinstance(init, dict) else None
    fin_d = fin if isinstance(fin, dict) else None
    return init_d, fin_d
