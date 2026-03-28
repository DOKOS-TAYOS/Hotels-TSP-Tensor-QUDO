"""Tests for sample histogram and energy-trajectory metrics in data_analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from data_analysis.metrics import (
    first_step_within_epsilon_of_ref,
    normalized_energy_auc,
    _histogram_sample_mass_split,
)
from data_analysis.optimal_sample_mass import (
    histogram_key_hamming_distance,
    histogram_mass_near_center,
    histogram_shannon_entropy,
    histogram_top_k_mass,
)


def test_histogram_shannon_entropy_uniform() -> None:
    n = 8
    hist = {f"k{i}": 1 for i in range(n)}
    h = histogram_shannon_entropy(hist, base=math.e)
    assert h is not None
    assert math.isclose(float(h), math.log(n))


def test_histogram_top_k_mass() -> None:
    hist = {"a": 10, "b": 5, "c": 1}
    assert histogram_top_k_mass(hist, 1) == pytest.approx(10 / 16)
    assert histogram_top_k_mass(hist, 2) == pytest.approx(15 / 16)
    assert histogram_top_k_mass(hist, 5) == pytest.approx(1.0)


def test_histogram_key_hamming_distance_qubo() -> None:
    assert histogram_key_hamming_distance("010", "000", "qubo", 3) == 1
    assert histogram_key_hamming_distance("01", "000", "qubo", 3) is None


def test_histogram_mass_near_center() -> None:
    # n_cities=3 → QUBO keys length (n-1)^2 = 4
    hist = {"0101": 3, "0100": 1}
    m = histogram_mass_near_center(hist, "0101", "qubo", 3, max_hamming=1)
    assert m == pytest.approx(1.0)


def test_normalized_energy_auc_decreasing_linear() -> None:
    hist = [3.0, 2.0, 1.0]
    auc = normalized_energy_auc(hist, initial_energy=3.0)
    assert auc is not None
    s = np.array([(x - 1.0) / (3.0 - 1.0) for x in hist])
    s = np.clip(s, 0.0, 1.0)
    if hasattr(np, "trapezoid"):
        expected = float(np.trapezoid(s))
    else:
        expected = float(np.trapz(s))
    assert math.isclose(float(auc), expected, rel_tol=1e-9)


def test_first_step_within_epsilon_of_ref() -> None:
    h = [100.0, 2.005, 1.2]
    st = first_step_within_epsilon_of_ref(h, 2.0, rel_tol=0.01, abs_floor=0.05)
    assert st == 2


def test_histogram_sample_mass_split_invalid_only() -> None:
    inst = {
        "n_cities": 4,
        "precedences": [[0, 1]],
        "prices_hotels": [[1.0, 2.0], [3.0, 4.0]],
        "prices_travels": [
            [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
            [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
            [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        ],
    }
    split = _histogram_sample_mass_split({"": 1, "not-a-key": 2}, inst)
    assert split is not None
    f, i, inv = split
    assert f == 0.0
    assert i == 0.0
    assert inv == 1.0
